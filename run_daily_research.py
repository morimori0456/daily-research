#!/usr/bin/env python3
"""Run scheduled daily Antigravity web research prompts and publish HTML reports."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import html
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable
from urllib.parse import urlsplit

try:
    from zoneinfo import ZoneInfo
    JST = ZoneInfo("Asia/Tokyo")
except Exception:  # pragma: no cover - fallback if tzdata is unavailable
    JST = dt.timezone(dt.timedelta(hours=9), name="Asia/Tokyo")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS_DIR = BASE_DIR / "prompts"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "output"
DEFAULT_CSS_PATH = BASE_DIR / "document.css"
DEFAULT_TIMEOUT_SEC = 1800
MAX_RESPONSE_BYTES = 8 * 1024 * 1024

DENIED_ACTIONS = [
    "command(*)",
    "read_file(*)",
    "write_file(*)",
    "mcp(*)",
    "execute_url(*)",
    "download_url(*)",
]
ALLOWED_ACTIONS = ["search_web(*)", "read_url(*)"]
ALLOWED_ENV_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)

HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HR_RE = re.compile(r"^ {0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$")
UL_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
OL_RE = re.compile(r"^(\s*)(\d+)\.\s+(.*)$")
BQ_RE = re.compile(r"^ {0,3}>\s?(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?\s*$")
CODE_SPAN_RE = re.compile(r"(`+)([^`]|[^`].*?[^`])\1")
BARE_URL_RE = re.compile(r"(https?://[^\s<>\"'`]+)")


@dataclass(frozen=True)
class PromptSpec:
    """Metadata and content for a single research prompt file."""

    slug: str
    prompt_id: str
    title: str
    path: Path
    text: str


def _now_jst() -> dt.datetime:
    return dt.datetime.now(JST)


def _atomic_write_text(path: Path, content: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp = Path(fh.name)
        fh.write(content)
    try:
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _extract_prompt_title(text: str, slug: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^#{1,6}\s+", "", line).strip()
        if cleaned:
            return cleaned[:120]
    return slug


def discover_prompts(prompts_dir: Path) -> list[PromptSpec]:
    """Load and sort prompt files (*.txt) by filename."""
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory does not exist: {prompts_dir}")
    specs: list[PromptSpec] = []
    for path in sorted(prompts_dir.glob("*.txt"), key=lambda p: p.name):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        slug = path.stem
        match = re.match(r"^(\d+)(?:-|$)", slug)
        prompt_id = match.group(1) if match else slug
        title = _extract_prompt_title(text, slug)
        specs.append(PromptSpec(slug=slug, prompt_id=prompt_id, title=title, path=path, text=text))
    if not specs:
        raise FileNotFoundError(f"No non-empty prompt files found in {prompts_dir}")
    return specs


def _matches_only_token(spec: PromptSpec, token: str) -> bool:
    if spec.slug == token or spec.path.name == token or spec.slug.startswith(token):
        return True
    if token.isdigit() and spec.prompt_id.isdigit() and int(token) == int(spec.prompt_id):
        return True
    return False


def filter_prompts(
    prompts: list[PromptSpec],
    only: str | None = None,
    limit: int | None = None,
) -> list[PromptSpec]:
    """Filter prompts by --only id/prefix and --limit."""
    selected = prompts
    if only is not None:
        tokens = [part.strip() for part in only.split(",") if part.strip()]
        if not tokens:
            raise ValueError("--only requires a non-empty id or prefix")
        selected = [p for p in prompts if any(_matches_only_token(p, tok) for tok in tokens)]
        if not selected:
            raise ValueError(f"No prompts matched --only={only!r}")
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be >= 0")
        selected = selected[:limit]
    return selected


def build_agy_settings() -> dict[str, Any]:
    """Return strict web-research-only settings for Antigravity CLI."""
    return {
        "toolPermission": "strict",
        "agentMode": "plan",
        "enableTerminalSandbox": True,
        "allowNonWorkspaceAccess": False,
        "historySize": 0,
        "permissions": {
            "deny": list(DENIED_ACTIONS),
            "allow": list(ALLOWED_ACTIONS),
        },
    }


def prepare_isolated_home(home: Path, source_cli_dir: Path | None = None) -> tuple[Path, Path]:
    """Create a private disposable HOME containing only OAuth token, bin/builtin, and strict settings."""
    source = source_cli_dir or (Path.home() / ".gemini" / "antigravity-cli")
    token = source / "antigravity-oauth-token"
    if not token.is_file() or token.is_symlink():
        raise RuntimeError(
            "Antigravity CLI is not signed in; run agy and sign in before running daily research"
        )

    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(home, 0o700)

    gemini_dir = home / ".gemini"
    cli_home = gemini_dir / "antigravity-cli"
    gemini_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    cli_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(gemini_dir, 0o700)
    os.chmod(cli_home, 0o700)

    isolated_token = cli_home / "antigravity-oauth-token"
    shutil.copyfile(token, isolated_token)
    os.chmod(isolated_token, 0o600)

    for name in ("bin", "builtin"):
        src = source / name
        if src.is_dir() and not src.is_symlink():
            shutil.copytree(src, cli_home / name, symlinks=False)

    (gemini_dir / "config").mkdir(mode=0o700, parents=True, exist_ok=True)
    for rel in (".config", ".cache", ".local", ".local/share", ".local/state", "tmp", "workspace"):
        p = home / rel
        p.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(p, 0o700)

    settings_path = home / "settings.json"
    settings_path.write_text(
        json.dumps(build_agy_settings(), separators=(",", ":")),
        encoding="utf-8",
    )
    os.chmod(settings_path, 0o600)
    return settings_path, home / "workspace"


def _terminate_group(proc: subprocess.Popen[str]) -> None:
    """Terminate the entire process group of a subprocess."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if proc.poll() is None:
        proc.wait()


def build_research_prompt(prompt_text: str, run_date: str) -> str:
    """Wrap the prompt with reference date and web-only Markdown constraints."""
    return (
        f"基準日: {run_date} (Asia/Tokyo)\n"
        "制約: 利用可能なツールは Web 検索・URL 閲覧（search_web / read_url）のみです。"
        "ローカルファイル参照やコマンド実行は行わず、調査結果を日本語の Markdown 形式で出力してください。\n\n"
        f"{prompt_text}"
    )


def _interrupt_handler(signum: int, _frame: Any) -> None:
    raise InterruptedError(f"research runner interrupted by signal {signum}")


def run_agy_prompt(
    prompt_text: str,
    run_date: str,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    agy_bin: str | None = None,
    source_cli_dir: Path | None = None,
) -> str:
    """Execute one research prompt via agy inside an isolated disposable HOME."""
    agy = agy_bin or shutil.which("agy")
    if not agy:
        raise RuntimeError("agy CLI not found in PATH")

    full_prompt = build_research_prompt(prompt_text, run_date)
    with tempfile.TemporaryDirectory(prefix="daily-research-agy-") as tmp_dir:
        home = Path(tmp_dir) / "home"
        settings_path, workspace = prepare_isolated_home(home, source_cli_dir=source_cli_dir)
        env = {key: os.environ[key] for key in ALLOWED_ENV_KEYS if key in os.environ}
        env.update(
            {
                "HOME": str(home),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_CACHE_HOME": str(home / ".cache"),
                "XDG_DATA_HOME": str(home / ".local" / "share"),
                "XDG_STATE_HOME": str(home / ".local" / "state"),
                "TMPDIR": str(home / "tmp"),
                "AGY_CLI_DISABLE_AUTO_UPDATE": "true",
            }
        )
        argv = [
            agy,
            "--settings",
            str(settings_path),
            "--sandbox",
            "--mode",
            "plan",
            "--disable-slash-commands",
            "--output-format",
            "json",
            "--print-timeout",
            f"{timeout_sec}s",
            f"--print={full_prompt}",
        ]
        proc: subprocess.Popen[str] | None = None
        is_main_thread = threading.current_thread() is threading.main_thread()
        prev_sigterm = signal.getsignal(signal.SIGTERM) if is_main_thread else None
        prev_sigint = signal.getsignal(signal.SIGINT) if is_main_thread else None
        if is_main_thread:
            signal.signal(signal.SIGTERM, _interrupt_handler)
            signal.signal(signal.SIGINT, _interrupt_handler)
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(workspace),
                env=env,
                start_new_session=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
            except subprocess.TimeoutExpired as exc:
                _terminate_group(proc)
                raise TimeoutError(f"agy research timed out after {timeout_sec} seconds") from exc
        finally:
            if proc is not None:
                _terminate_group(proc)
                if proc.stdout is not None:
                    proc.stdout.close()
                if proc.stderr is not None:
                    proc.stderr.close()
            if is_main_thread:
                signal.signal(signal.SIGTERM, prev_sigterm)
                signal.signal(signal.SIGINT, prev_sigint)

        if proc.returncode != 0:
            err_snippet = (stderr or "").strip().splitlines()
            detail = f": {err_snippet[-1][:200]}" if err_snippet else ""
            raise RuntimeError(f"agy exited with status {proc.returncode}{detail}")

        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("agy did not return valid JSON output") from exc

        if not isinstance(envelope, dict) or envelope.get("status") != "SUCCESS":
            status = envelope.get("status", "invalid") if isinstance(envelope, dict) else "invalid"
            raise RuntimeError(f"agy returned non-success status: {status}")

        if envelope.get("denied_actions"):
            raise RuntimeError("agy requested a denied tool action")

        response = envelope.get("response")
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError("agy returned an empty response")

        if len(response.encode("utf-8", errors="replace")) > MAX_RESPONSE_BYTES:
            raise RuntimeError("agy response exceeded size limit")

        return response.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def is_safe_http_url(url: str) -> bool:
    """Allow only absolute http/https URLs with a valid netloc."""
    candidate = url.strip()
    if not candidate or any(ch.isspace() or ord(ch) < 32 for ch in candidate):
        return False
    parsed = urlsplit(candidate)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _strip_inline_markdown(text: str) -> str:
    """Remove common inline Markdown markers for plain text labels."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__|\*|_|~~)", "", text)
    return text.strip()


def _render_bare_urls_and_text(segment: str) -> str:
    out: list[str] = []
    pos = 0
    for match in BARE_URL_RE.finditer(segment):
        start, end = match.span()
        if start > pos:
            out.append(html.escape(segment[pos:start], quote=True))
        raw_url = match.group(1)
        trailing = ""
        while raw_url and raw_url[-1] in ".,;:!?)>]":
            if raw_url[-1] == ")" and raw_url.count("(") >= raw_url.count(")"):
                break
            trailing = raw_url[-1] + trailing
            raw_url = raw_url[:-1]
        if is_safe_http_url(raw_url):
            esc_url = html.escape(raw_url, quote=True)
            out.append(f'<a href="{esc_url}" rel="noopener noreferrer">{esc_url}</a>')
        else:
            out.append(html.escape(raw_url, quote=True))
        if trailing:
            out.append(html.escape(trailing, quote=True))
        pos = end
    if pos < len(segment):
        out.append(html.escape(segment[pos:], quote=True))
    return "".join(out)


def _format_emphasis(segment: str) -> str:
    """Render bold, italic, and strikethrough on plain text with URL auto-linking."""
    pattern = re.compile(
        r"(\*\*(?=\S)(?:.*?\S)?\*\*|__(?=\S)(?:.*?\S)?__"
        r"|~~(?=\S)(?:.*?\S)?~~"
        r"|(?<![A-Za-z0-9])\*(?=\S)(?:[^*]*?\S)?\*(?![A-Za-z0-9])"
        r"|(?<![A-Za-z0-9_])_(?=\S)(?:[^_]*?\S)?_(?![A-Za-z0-9_]))"
    )
    out: list[str] = []
    pos = 0
    for match in pattern.finditer(segment):
        start, end = match.span()
        if start > pos:
            out.append(_render_bare_urls_and_text(segment[pos:start]))
        token = match.group(0)
        if token.startswith("**") or token.startswith("__"):
            out.append(f"<strong>{_format_emphasis(token[2:-2])}</strong>")
        elif token.startswith("~~"):
            out.append(f"<del>{_format_emphasis(token[2:-2])}</del>")
        else:
            out.append(f"<em>{_format_emphasis(token[1:-1])}</em>")
        pos = end
    if pos < len(segment):
        out.append(_render_bare_urls_and_text(segment[pos:]))
    return "".join(out)


def _find_closing_bracket(text: str, start: int) -> int:
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _find_closing_paren(text: str, start: int) -> int:
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _parse_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")].strip()
    m = re.match(r'^(\S+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?$', target)
    return m.group(1) if m else target


def _render_non_code_inline(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    buf: list[str] = []

    def flush_buf() -> None:
        if buf:
            out.append(_format_emphasis("".join(buf)))
            buf.clear()

    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n and text[i + 1] in r"\`*_{}[]()#+-.!|<>~":
            buf.append(text[i + 1])
            i += 2
            continue
        if ch == "<":
            close = text.find(">", i + 1)
            if close != -1:
                candidate = text[i + 1 : close].strip()
                if is_safe_http_url(candidate):
                    flush_buf()
                    esc_url = html.escape(candidate, quote=True)
                    out.append(f'<a href="{esc_url}" rel="noopener noreferrer">{esc_url}</a>')
                    i = close + 1
                    continue
        if ch == "[":
            close_bracket = _find_closing_bracket(text, i + 1)
            if close_bracket != -1 and close_bracket + 1 < n and text[close_bracket + 1] == "(":
                close_paren = _find_closing_paren(text, close_bracket + 2)
                if close_paren != -1:
                    label = text[i + 1 : close_bracket]
                    raw_target = text[close_bracket + 2 : close_paren]
                    url = _parse_link_target(raw_target)
                    flush_buf()
                    rendered_label = _render_non_code_inline(label)
                    if is_safe_http_url(url):
                        esc_url = html.escape(url, quote=True)
                        out.append(f'<a href="{esc_url}" rel="noopener noreferrer">{rendered_label}</a>')
                    else:
                        out.append(f"{rendered_label} ({html.escape(url, quote=True)})")
                    i = close_paren + 1
                    continue
        buf.append(ch)
        i += 1
    flush_buf()
    return "".join(out)


def render_inline(text: str) -> str:
    """Render inline Markdown safely with HTML escaping and URL scheme validation."""
    out: list[str] = []
    pos = 0
    for match in CODE_SPAN_RE.finditer(text):
        start, end = match.span()
        if start > pos:
            out.append(_render_non_code_inline(text[pos:start]))
        code_body = match.group(2).strip()
        out.append(f"<code>{html.escape(code_body, quote=True)}</code>")
        pos = end
    if pos < len(text):
        out.append(_render_non_code_inline(text[pos:]))
    return "".join(out)


def _split_table_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells: list[str] = []
    cur: list[str] = []
    in_code = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "`":
            in_code = not in_code
            cur.append(ch)
            i += 1
            continue
        if ch == "|" and not in_code:
            cells.append("".join(cur).strip())
            cur.clear()
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _is_table_start(lines: list[str], idx: int) -> bool:
    if idx + 1 >= len(lines):
        return False
    header = lines[idx]
    sep = lines[idx + 1]
    if "|" not in header:
        return False
    if not TABLE_SEP_RE.match(sep):
        return False
    header_cells = _split_table_row(header)
    sep_cells = _split_table_row(sep)
    return len(header_cells) >= 1 and len(header_cells) == len(sep_cells)


def _is_block_boundary(lines: list[str], idx: int) -> bool:
    line = lines[idx]
    if not line.strip():
        return True
    if HEADING_RE.match(line) or FENCE_RE.match(line) or HR_RE.match(line) or BQ_RE.match(line):
        return True
    if UL_RE.match(line) or OL_RE.match(line):
        return True
    if _is_table_start(lines, idx):
        return True
    return False


def _sanitize_lang(raw_info: str) -> str:
    token = raw_info.strip().split()[0] if raw_info.strip() else ""
    cleaned = re.sub(r"[^A-Za-z0-9_+-]", "", token)
    return cleaned.lower()


def _parse_list(lines: list[str], start: int) -> tuple[str, int]:
    """Parse nested unordered/ordered Markdown lists starting at `start`."""

    def match_item(line: str) -> tuple[int, str, str] | None:
        m_ul = UL_RE.match(line)
        if m_ul:
            indent = len(m_ul.group(1).replace("\t", "    "))
            return indent, "ul", m_ul.group(2)
        m_ol = OL_RE.match(line)
        if m_ol:
            indent = len(m_ol.group(1).replace("\t", "    "))
            return indent, "ol", m_ol.group(3)
        return None

    first = match_item(lines[start])
    assert first is not None
    base_indent, list_tag, _ = first
    items_html: list[str] = []
    idx = start

    while idx < len(lines):
        item = match_item(lines[idx])
        if item is None:
            break
        indent, tag, content = item
        if indent < base_indent or (indent == base_indent and tag != list_tag):
            break
        if indent > base_indent:
            nested_html, idx = _parse_list(lines, idx)
            if items_html:
                items_html[-1] = items_html[-1][:-5] + "\n" + nested_html + "\n</li>"
            else:
                items_html.append(f"<li>{nested_html}</li>")
            continue

        item_lines = [content.strip()]
        idx += 1
        while idx < len(lines):
            nxt = lines[idx]
            if not nxt.strip():
                # Peek past blank lines to see if a nested list or same-list item continues
                look = idx
                while look < len(lines) and not lines[look].strip():
                    look += 1
                if look < len(lines):
                    nxt_item = match_item(lines[look])
                    if nxt_item and nxt_item[0] > base_indent:
                        idx = look
                        nested_html, idx = _parse_list(lines, idx)
                        item_lines.append(nested_html)
                        continue
                break
            nxt_item = match_item(nxt)
            if nxt_item is not None:
                if nxt_item[0] > base_indent:
                    nested_html, idx = _parse_list(lines, idx)
                    item_lines.append(nested_html)
                    continue
                break
            if HEADING_RE.match(nxt) or FENCE_RE.match(nxt) or HR_RE.match(nxt) or BQ_RE.match(nxt):
                break
            if _is_table_start(lines, idx):
                break
            item_lines.append(nxt.strip())
            idx += 1

        # Render inline text parts while preserving nested list blocks
        rendered_parts: list[str] = []
        text_acc: list[str] = []
        for part in item_lines:
            if part.startswith("<ul>") or part.startswith("<ol>"):
                if text_acc:
                    rendered_parts.append(render_inline(" ".join(text_acc)))
                    text_acc.clear()
                rendered_parts.append("\n" + part)
            else:
                text_acc.append(part)
        if text_acc:
            rendered_parts.append(render_inline(" ".join(text_acc)))
        items_html.append(f"<li>{''.join(rendered_parts)}</li>")

    return f"<{list_tag}>\n" + "\n".join(items_html) + f"\n</{list_tag}>", idx


def _compute_heading_plan(lines: list[str]) -> tuple[int | None, int]:
    """Determine optional leading title line index and heading level offset."""
    headings: list[tuple[int, int]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    for idx, line in enumerate(lines):
        m_fence = FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and not m_fence.group(2).strip():
                in_fence = False
            continue
        if in_fence:
            continue
        m_head = HEADING_RE.match(line)
        if m_head:
            headings.append((idx, len(m_head.group(1))))

    if not headings:
        return None, 0

    h1_indices = [idx for idx, lvl in headings if lvl == 1]
    h2_indices = [idx for idx, lvl in headings if lvl == 2]
    if len(h1_indices) == 1 and h1_indices[0] == headings[0][0] and h2_indices:
        return h1_indices[0], 0
    if h1_indices:
        return None, 1
    min_level = min(lvl for _, lvl in headings)
    if min_level >= 3:
        return None, 2 - min_level
    return None, 0


def _make_heading_id(text: str, index: int, used: set[str]) -> str:
    ascii_slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    base = f"sec-{index}-{ascii_slug[:40]}" if ascii_slug else f"sec-{index}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def convert_markdown_body(
    md_text: str,
    used_ids: set[str] | None = None,
    apply_heading_plan: bool = True,
) -> tuple[str, list[tuple[str, str]], str | None, list[str]]:
    """Convert Markdown text into sanitized HTML body, h2 TOC entries, extracted title, and summary bullets."""
    normalized = md_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    strip_h1_idx, level_shift = _compute_heading_plan(lines) if apply_heading_plan else (None, 0)

    if used_ids is None:
        used_ids = set()
    toc_entries: list[tuple[str, str]] = []
    extracted_title: str | None = None
    summary_bullets: list[str] = []
    blocks: list[str] = []

    idx = 0
    n = len(lines)
    while idx < n:
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        m_fence = FENCE_RE.match(line)
        if m_fence:
            marker = m_fence.group(1)
            fence_char = marker[0]
            fence_len = len(marker)
            lang = _sanitize_lang(m_fence.group(2))
            idx += 1
            code_lines: list[str] = []
            while idx < n:
                end_m = FENCE_RE.match(lines[idx])
                if (
                    end_m
                    and end_m.group(1)[0] == fence_char
                    and len(end_m.group(1)) >= fence_len
                    and not end_m.group(2).strip()
                ):
                    idx += 1
                    break
                code_lines.append(lines[idx])
                idx += 1
            cls_attr = f' class="language-{lang}"' if lang else ""
            code_escaped = html.escape("\n".join(code_lines), quote=True)
            blocks.append(f"<pre><code{cls_attr}>{code_escaped}</code></pre>")
            continue

        m_head = HEADING_RE.match(line)
        if m_head:
            raw_level = len(m_head.group(1))
            raw_text = m_head.group(2).strip()
            if strip_h1_idx is not None and idx == strip_h1_idx:
                extracted_title = _strip_inline_markdown(raw_text)
                idx += 1
                continue
            level = max(2, min(6, raw_level + level_shift))
            rendered_head = render_inline(raw_text)
            if level == 2:
                plain_head = _strip_inline_markdown(raw_text) or f"Section {len(toc_entries) + 1}"
                sec_id = _make_heading_id(plain_head, len(toc_entries) + 1, used_ids)
                toc_entries.append((sec_id, plain_head))
                blocks.append(f'<h2 id="{html.escape(sec_id, quote=True)}">{rendered_head}</h2>')
            else:
                blocks.append(f"<h{level}>{rendered_head}</h{level}>")
            idx += 1
            continue

        if HR_RE.match(line):
            blocks.append("<hr>")
            idx += 1
            continue

        if BQ_RE.match(line):
            quote_lines: list[str] = []
            while idx < n:
                m_bq = BQ_RE.match(lines[idx])
                if not m_bq:
                    break
                quote_lines.append(m_bq.group(1))
                idx += 1
            inner_html, _, _, _ = convert_markdown_body(
                "\n".join(quote_lines), used_ids=used_ids, apply_heading_plan=False
            )
            blocks.append(f"<blockquote>\n{inner_html}\n</blockquote>")
            continue

        if _is_table_start(lines, idx):
            headers = _split_table_row(lines[idx])
            idx += 2
            body_rows: list[list[str]] = []
            while idx < n and lines[idx].strip() and "|" in lines[idx]:
                if HEADING_RE.match(lines[idx]) or FENCE_RE.match(lines[idx]):
                    break
                row_cells = _split_table_row(lines[idx])
                if len(row_cells) < len(headers):
                    row_cells.extend([""] * (len(headers) - len(row_cells)))
                elif len(row_cells) > len(headers):
                    row_cells = row_cells[: len(headers)]
                body_rows.append(row_cells)
                idx += 1
            th_html = "".join(f"<th>{render_inline(cell)}</th>" for cell in headers)
            tr_html = "".join(
                "<tr>" + "".join(f"<td>{render_inline(cell)}</td>" for cell in row) + "</tr>"
                for row in body_rows
            )
            blocks.append(
                f"<table>\n<thead><tr>{th_html}</tr></thead>\n<tbody>{tr_html}</tbody>\n</table>"
            )
            continue

        if UL_RE.match(line) or OL_RE.match(line):
            list_html, idx = _parse_list(lines, idx)
            blocks.append(list_html)
            if not summary_bullets:
                for raw_li in re.findall(r"<li>(.*?)</li>", list_html, flags=re.DOTALL):
                    clean_li = re.sub(r"<[^>]+>", "", html.unescape(raw_li)).strip()
                    clean_li = re.sub(r"\s+", " ", clean_li)
                    if clean_li:
                        summary_bullets.append(clean_li[:180])
                    if len(summary_bullets) >= 4:
                        break
            continue

        para_lines: list[str] = []
        while idx < n and not _is_block_boundary(lines, idx):
            para_lines.append(lines[idx].strip())
            idx += 1
        if para_lines:
            blocks.append(f"<p>{render_inline(' '.join(para_lines))}</p>")

    return "\n".join(blocks), toc_entries, extracted_title, summary_bullets


def render_report_html(
    markdown_text: str,
    spec: PromptSpec,
    run_date: str,
    css_path: Path = DEFAULT_CSS_PATH,
    generated_at: str | None = None,
) -> str:
    """Render a standalone CSP-safe HTML report using the design system stylesheet."""
    css_text = css_path.read_text(encoding="utf-8")
    if "</style" in css_text.lower():
        raise ValueError("CSS file contains forbidden closing style tag")

    body_html, toc_entries, extracted_title, summary_bullets = convert_markdown_body(markdown_text)
    if not toc_entries:
        sec_id = "sec-1"
        toc_entries = [(sec_id, "調査結果")]
        body_html = f'<h2 id="{sec_id}">調査結果</h2>\n' + body_html

    doc_title = extracted_title or spec.title or spec.slug
    timestamp = generated_at or _now_jst().isoformat(timespec="seconds")

    toc_items = "".join(
        f'<li><a href="#{html.escape(sec_id, quote=True)}">{html.escape(label)}</a></li>'
        for sec_id, label in toc_entries
    )
    summary_items = [
        f"調査日: {run_date}（基準プロンプト: {spec.path.name}）",
        f"構成セクション数: {len(toc_entries)}",
    ]
    for bullet in summary_bullets[:3]:
        summary_items.append(bullet)
    summary_html = "".join(f"<li>{html.escape(item)}</li>" for item in summary_items)

    esc_title = html.escape(doc_title)
    esc_slug = html.escape(spec.slug)
    esc_date = html.escape(run_date)
    esc_ts = html.escape(timestamp)
    esc_lede = html.escape(
        f"{run_date} に Antigravity Web 検索（search_web / read_url）で自動収集した "
        f"「{doc_title}」の調査レポートです。"
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="ja">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<title>{esc_title} ({esc_date})</title>\n"
        f"<style>\n{css_text}\n</style>\n"
        "</head>\n"
        "<body>\n"
        '<div class="mb-page">\n'
        '<div class="mb-hero">\n'
        '<header class="mb-header">\n'
        "  <div>\n"
        f"    <h1>{esc_title}</h1>\n"
        f'    <p class="mb-meta-line">{esc_date} · {esc_slug} · 生成: {esc_ts}</p>\n'
        "  </div>\n"
        "</header>\n"
        f'<p class="mb-lede">{esc_lede}</p>\n'
        '<nav class="mb-toc" aria-label="目次">\n'
        f"  <ol>{toc_items}</ol>\n"
        "</nav>\n"
        '<div class="mb-summary">\n'
        f"  <ul>{summary_html}</ul>\n"
        "</div>\n"
        "</div>\n"
        '<div class="mb-wrap">\n'
        '<main class="mb-main">\n'
        f"{body_html}\n"
        "</main>\n"
        "</div>\n"
        f'<footer class="mb-footer">Daily Research · {esc_date} · {esc_slug}</footer>\n'
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )


def load_state(state_path: Path, run_date: str) -> dict[str, Any]:
    """Load daily execution state, initializing if absent."""
    if not state_path.is_file():
        return {"date": run_date, "updated_at": _now_jst().isoformat(timespec="seconds"), "prompts": {}}
    data = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("prompts"), dict):
        return {"date": run_date, "updated_at": _now_jst().isoformat(timespec="seconds"), "prompts": {}}
    return data


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    """Persist daily execution state atomically."""
    state["updated_at"] = _now_jst().isoformat(timespec="seconds")
    _atomic_write_text(state_path, json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def run_daily(
    *,
    prompts_dir: Path,
    output_root: Path,
    dashboard_reports_dir: Path,
    css_path: Path = DEFAULT_CSS_PATH,
    run_date: str | None = None,
    only: str | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    runner: Callable[[str, str, int], str] | None = None,
) -> int:
    """Execute scheduled daily research prompts with idempotent state tracking."""
    if timeout_sec <= 0:
        raise ValueError("--timeout must be > 0")

    effective_date = run_date or _now_jst().date().isoformat()
    compact_date = effective_date.replace("-", "")
    prompts = discover_prompts(prompts_dir)
    selected = filter_prompts(prompts, only=only, limit=limit)

    day_output_dir = output_root / effective_date
    state_path = day_output_dir / "state.json"
    state = load_state(state_path, effective_date)

    if dry_run:
        plan = []
        for spec in selected:
            existing = state["prompts"].get(spec.slug, {})
            would_skip = existing.get("status") == "completed" and not force
            md_path = day_output_dir / f"{compact_date}-{spec.slug}.md"
            html_path = day_output_dir / f"{compact_date}-{spec.slug}.html"
            dash_path = dashboard_reports_dir / f"{compact_date}-{spec.slug}.html"
            plan.append(
                {
                    "slug": spec.slug,
                    "prompt_file": str(spec.path),
                    "action": "skip" if would_skip else "run",
                    "existing_status": existing.get("status"),
                    "timeout_sec": timeout_sec,
                    "markdown_path": str(md_path),
                    "html_path": str(html_path),
                    "dashboard_html_path": str(dash_path),
                }
            )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "date": effective_date,
                    "force": force,
                    "timeout_sec": timeout_sec,
                    "settings": build_agy_settings(),
                    "prompts": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    day_output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_reports_dir.mkdir(parents=True, exist_ok=True)
    execute_fn = runner or (lambda text, d, t: run_agy_prompt(text, d, timeout_sec=t))

    failures = 0
    for spec in selected:
        existing = state["prompts"].get(spec.slug, {})
        if existing.get("status") == "completed" and not force:
            print(f"[skip] {spec.slug} (already completed for {effective_date})")
            continue

        started_at = _now_jst().isoformat(timespec="seconds")
        md_path = day_output_dir / f"{compact_date}-{spec.slug}.md"
        html_path = day_output_dir / f"{compact_date}-{spec.slug}.html"
        dash_path = dashboard_reports_dir / f"{compact_date}-{spec.slug}.html"
        print(f"[run] {spec.slug} (timeout={timeout_sec}s)")

        try:
            markdown_out = execute_fn(spec.text, effective_date, timeout_sec)
            generated_at = _now_jst().isoformat(timespec="seconds")
            html_out = render_report_html(
                markdown_out,
                spec=spec,
                run_date=effective_date,
                css_path=css_path,
                generated_at=generated_at,
            )
            _atomic_write_text(md_path, markdown_out)
            _atomic_write_text(html_path, html_out)
            _atomic_write_text(dash_path, html_out)
            state["prompts"][spec.slug] = {
                "prompt_file": spec.path.name,
                "slug": spec.slug,
                "status": "completed",
                "started_at": started_at,
                "finished_at": _now_jst().isoformat(timespec="seconds"),
                "markdown_path": str(md_path),
                "html_path": str(html_path),
                "dashboard_html_path": str(dash_path),
                "error": None,
            }
            save_state(state_path, state)
            print(f"[ok] {spec.slug} -> {dash_path}")
        except (InterruptedError, KeyboardInterrupt) as exc:
            state["prompts"][spec.slug] = {
                "prompt_file": spec.path.name,
                "slug": spec.slug,
                "status": "failed",
                "started_at": started_at,
                "finished_at": _now_jst().isoformat(timespec="seconds"),
                "markdown_path": str(md_path),
                "html_path": str(html_path),
                "dashboard_html_path": str(dash_path),
                "error": str(exc) or "interrupted",
            }
            save_state(state_path, state)
            print(f"[interrupted] {spec.slug}: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            failures += 1
            state["prompts"][spec.slug] = {
                "prompt_file": spec.path.name,
                "slug": spec.slug,
                "status": "failed",
                "started_at": started_at,
                "finished_at": _now_jst().isoformat(timespec="seconds"),
                "markdown_path": str(md_path),
                "html_path": str(html_path),
                "dashboard_html_path": str(dash_path),
                "error": str(exc),
            }
            save_state(state_path, state)
            print(f"[failed] {spec.slug}: {exc}", file=sys.stderr)

    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show execution plan and settings without calling agy or writing files",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Run only prompts matching id or prefix (comma-separated supported)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of prompts to execute",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run prompts even if already marked completed today",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"Timeout per prompt in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override target date YYYY-MM-DD (default: today in Asia/Tokyo)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.date is not None:
        dt.date.fromisoformat(args.date)
    dashboard_root = Path(os.environ.get("DASHBOARD_DATA", Path.home() / ".local/share/dashboard"))
    prompts_dir = Path(os.environ.get("DAILY_RESEARCH_PROMPTS_DIR", DEFAULT_PROMPTS_DIR))
    output_root = Path(os.environ.get("DAILY_RESEARCH_OUTPUT_DIR", DEFAULT_OUTPUT_ROOT))
    dashboard_reports_dir = Path(
        os.environ.get("DASHBOARD_REPORTS_DIR", dashboard_root / "reports" / "daily-research")
    )
    try:
        return run_daily(
            prompts_dir=prompts_dir,
            output_root=output_root,
            dashboard_reports_dir=dashboard_reports_dir,
            css_path=DEFAULT_CSS_PATH,
            run_date=args.date,
            only=args.only,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
            timeout_sec=args.timeout,
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
