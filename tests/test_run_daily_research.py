"""Tests for daily_research/run_daily_research.py."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

DAILY_RESEARCH_DIR = Path(__file__).resolve().parents[1]
if str(DAILY_RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(DAILY_RESEARCH_DIR))

import run_daily_research as rdr  # noqa: E402


class PromptDiscoveryTests(unittest.TestCase):
    def test_discover_real_prompts_in_order(self) -> None:
        prompts = rdr.discover_prompts(DAILY_RESEARCH_DIR / "prompts")
        self.assertEqual(
            [p.slug for p in prompts],
            [
                "01-latest-ai-cloud-and-local-llm-cases-tips",
                "02-emerging-dev-tools-trending",
                "03-autonomous-driving-paper-trends",
                "04-llm-agent-context-harness-engineering-papers",
            ],
        )
        for p in prompts:
            self.assertNotIn("\r", p.text)
            self.assertTrue(len(p.text) > 100)

    def test_filter_only_and_limit(self) -> None:
        prompts = rdr.discover_prompts(DAILY_RESEARCH_DIR / "prompts")
        self.assertEqual(
            [p.prompt_id for p in rdr.filter_prompts(prompts, only="02")],
            ["02"],
        )
        self.assertEqual(
            [p.prompt_id for p in rdr.filter_prompts(prompts, only="3")],
            ["03"],
        )
        self.assertEqual(
            [p.prompt_id for p in rdr.filter_prompts(prompts, only="01,04")],
            ["01", "04"],
        )
        self.assertEqual(
            [p.prompt_id for p in rdr.filter_prompts(prompts, limit=2)],
            ["01", "02"],
        )
        with self.assertRaises(ValueError):
            rdr.filter_prompts(prompts, only="99-nonexistent")
        with self.assertRaises(ValueError):
            rdr.filter_prompts(prompts, limit=-1)


class MarkdownAndHtmlTests(unittest.TestCase):
    def _sample_spec(self) -> rdr.PromptSpec:
        return rdr.PromptSpec(
            slug="01-sample-topic",
            prompt_id="01",
            title="サンプルトピック調査",
            path=Path("/tmp/01-sample-topic.txt"),
            text="調査してください",
        )

    def test_html_structure_csp_and_markdown_elements(self) -> None:
        md = """# 週間 AI トレンド

## 全体トレンドの要約

- 重要項目1: **Local LLM** の推論最適化が進展
  - サブ項目: `vLLM` と `SGLang`
- 重要項目2: [公式ドキュメント](https://example.com/docs_(v2)) を参照

## 比較表

| 項目 | クラウド | ローカル |
| --- | --- | --- |
| 速度 | 高速 | VRAM依存 |
| 費用 | 従量課金 | 初期投資のみ |

### コード例

```python
print("<script>alert(1)</script>")
```

> 引用テスト: https://example.org/ref
"""
        html_doc = rdr.render_report_html(
            md,
            spec=self._sample_spec(),
            run_date="2026-11-14",
            css_path=DAILY_RESEARCH_DIR / "document.css",
            generated_at="2026-11-14T05:00:00+09:00",
        )

        # Required structural classes
        for cls in (
            "mb-page",
            "mb-hero",
            "mb-header",
            "mb-lede",
            "mb-toc",
            "mb-summary",
            "mb-wrap",
            "mb-main",
        ):
            self.assertIn(cls, html_doc)

        # Inline stylesheet present, no external stylesheet or scripts
        self.assertIn("<style>", html_doc)
        self.assertIn("--mb-bg: #FAF9F6;", html_doc)
        self.assertNotIn("<script", html_doc.lower())
        self.assertNotIn('type="module"', html_doc)
        self.assertNotIn("@font-face", html_doc)
        self.assertNotIn("localStorage", html_doc)
        self.assertNotIn("fetch(", html_doc)

        # TOC links match h2 IDs
        self.assertIn('href="#sec-1"', html_doc)
        self.assertIn('<h2 id="sec-1">全体トレンドの要約</h2>', html_doc)
        self.assertIn('href="#sec-2"', html_doc)
        self.assertIn('<h2 id="sec-2">比較表</h2>', html_doc)

        # Table, code block, blockquote, nested list, and safe links
        self.assertIn("<table>", html_doc)
        self.assertIn("<th>項目</th>", html_doc)
        self.assertIn("<td>従量課金</td>", html_doc)
        self.assertIn('<code class="language-python">', html_doc)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html_doc)
        self.assertIn("<blockquote>", html_doc)
        self.assertIn('href="https://example.com/docs_(v2)"', html_doc)
        self.assertIn('href="https://example.org/ref"', html_doc)

    def test_xss_and_unsafe_url_schemes_blocked(self) -> None:
        md = """## セキュリティ確認

<script>alert("xss")</script>
<img src=x onerror=alert(1)>

- [JSリンク](javascript:alert(1))
- [Dataリンク](data:text/html,<script>alert(1)</script>)
- [相対リンク](/etc/passwd)
- [正常リンク](https://example.com/safe?a=1&b=2)
"""
        html_doc = rdr.render_report_html(
            md,
            spec=self._sample_spec(),
            run_date="2026-11-14",
            css_path=DAILY_RESEARCH_DIR / "document.css",
        )
        self.assertNotIn("<script", html_doc.lower())
        self.assertNotIn("<img", html_doc.lower())
        self.assertNotIn('href="javascript:', html_doc.lower())
        self.assertNotIn('href="data:', html_doc.lower())
        self.assertNotIn('href="/etc/passwd"', html_doc)
        self.assertIn("&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;", html_doc)
        self.assertIn('href="https://example.com/safe?a=1&amp;b=2"', html_doc)


class IsolatedAgyRunnerTests(unittest.TestCase):
    def _create_fake_cli_dir(self, root: Path) -> Path:
        cli_dir = root / "source_cli"
        cli_dir.mkdir(parents=True)
        (cli_dir / "antigravity-oauth-token").write_text("secret-oauth-token", encoding="utf-8")
        (cli_dir / "bin").mkdir()
        (cli_dir / "bin" / "helper").write_text("#!/bin/sh\n", encoding="utf-8")
        (cli_dir / "builtin").mkdir()
        (cli_dir / "builtin" / "keep.txt").write_text("ok", encoding="utf-8")
        return cli_dir

    def test_prepare_isolated_home_permissions_and_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli_dir = self._create_fake_cli_dir(root)
            home = root / "isolated"
            settings_path, workspace = rdr.prepare_isolated_home(home, source_cli_dir=cli_dir)

            self.assertEqual(stat.S_IMODE(home.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(settings_path.stat().st_mode), 0o600)
            token_copy = home / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
            self.assertEqual(stat.S_IMODE(token_copy.stat().st_mode), 0o600)
            self.assertTrue((home / ".gemini" / "antigravity-cli" / "bin" / "helper").is_file())
            self.assertTrue((home / ".gemini" / "antigravity-cli" / "builtin" / "keep.txt").is_file())
            self.assertTrue(workspace.is_dir())

            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            self.assertEqual(settings["toolPermission"], "strict")
            self.assertEqual(settings["permissions"]["allow"], ["search_web(*)", "read_url(*)"])
            self.assertEqual(
                settings["permissions"]["deny"],
                [
                    "command(*)",
                    "read_file(*)",
                    "write_file(*)",
                    "mcp(*)",
                    "execute_url(*)",
                    "download_url(*)",
                ],
            )

    def test_prepare_isolated_home_rejects_symlink_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_secret = root / "real-token"
            real_secret.write_text("tok", encoding="utf-8")
            cli_dir = root / "source_cli"
            cli_dir.mkdir()
            (cli_dir / "antigravity-oauth-token").symlink_to(real_secret)
            with self.assertRaises(RuntimeError):
                rdr.prepare_isolated_home(root / "isolated", source_cli_dir=cli_dir)

    def test_run_agy_prompt_with_fake_binary_and_env_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli_dir = self._create_fake_cli_dir(root)
            fake_agy = root / "agy"
            record_file = root / "invocation.json"
            fake_agy.write_text(
                f"""#!/usr/bin/env python3
import json, os, sys
record = {{
    "argv": sys.argv[1:],
    "home": os.environ.get("HOME"),
    "disable_auto_update": os.environ.get("AGY_CLI_DISABLE_AUTO_UPDATE"),
    "leaked_secret": os.environ.get("GEMINI_API_KEY"),
    "cwd": os.getcwd(),
    "token_exists": os.path.isfile(os.path.join(os.environ["HOME"], ".gemini/antigravity-cli/antigravity-oauth-token")),
}}
with open({str(record_file)!r}, "w", encoding="utf-8") as fh:
    json.dump(record, fh)
print(json.dumps({{"status": "SUCCESS", "response": "## 調査結果\\n\\n- テスト項目"}}))
""",
                encoding="utf-8",
            )
            fake_agy.chmod(0o755)

            old_secret = os.environ.get("GEMINI_API_KEY")
            os.environ["GEMINI_API_KEY"] = "should-not-leak"
            try:
                out = rdr.run_agy_prompt(
                    "調査プロンプト",
                    run_date="2026-11-14",
                    timeout_sec=30,
                    agy_bin=str(fake_agy),
                    source_cli_dir=cli_dir,
                )
            finally:
                if old_secret is None:
                    os.environ.pop("GEMINI_API_KEY", None)
                else:
                    os.environ["GEMINI_API_KEY"] = old_secret

            self.assertIn("## 調査結果", out)
            record = json.loads(record_file.read_text(encoding="utf-8"))
            self.assertNotEqual(record["home"], str(Path.home()))
            self.assertEqual(record["disable_auto_update"], "true")
            self.assertIsNone(record["leaked_secret"])
            self.assertTrue(record["token_exists"])
            self.assertIn("--sandbox", record["argv"])
            self.assertIn("plan", record["argv"])

    def test_run_agy_prompt_rejects_denied_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli_dir = self._create_fake_cli_dir(root)
            fake_agy = root / "agy"
            fake_agy.write_text(
                """#!/usr/bin/env python3
import json
print(json.dumps({"status": "SUCCESS", "denied_actions": ["command(ls)"], "response": "ok"}))
""",
                encoding="utf-8",
            )
            fake_agy.chmod(0o755)
            with self.assertRaises(RuntimeError):
                rdr.run_agy_prompt(
                    "test",
                    run_date="2026-11-14",
                    timeout_sec=10,
                    agy_bin=str(fake_agy),
                    source_cli_dir=cli_dir,
                )

    def test_run_agy_prompt_timeout_terminates_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli_dir = self._create_fake_cli_dir(root)
            pid_file = root / "child.pid"
            fake_agy = root / "agy"
            fake_agy.write_text(
                f"""#!/usr/bin/env python3
import subprocess, time
child = subprocess.Popen(["sleep", "30"])
with open({str(pid_file)!r}, "w", encoding="utf-8") as fh:
    fh.write(str(child.pid))
time.sleep(30)
""",
                encoding="utf-8",
            )
            fake_agy.chmod(0o755)
            with self.assertRaises(TimeoutError):
                rdr.run_agy_prompt(
                    "test",
                    run_date="2026-11-14",
                    timeout_sec=1,
                    agy_bin=str(fake_agy),
                    source_cli_dir=cli_dir,
                )
            self.assertTrue(pid_file.is_file())
            child_pid = int(pid_file.read_text(encoding="utf-8").strip())
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)


class DailyRunnerEndToEndTests(unittest.TestCase):
    def _setup_workspace(self, root: Path) -> tuple[Path, Path, Path]:
        prompts_dir = root / "prompts"
        output_root = root / "output"
        dash_dir = root / "dashboard" / "reports" / "daily-research"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "01-alpha.txt").write_text("# Alpha 調査\n詳細を調べてください。", encoding="utf-8")
        (prompts_dir / "02-beta.txt").write_text("# Beta 調査\n詳細を調べてください。", encoding="utf-8")
        return prompts_dir, output_root, dash_dir

    def test_dry_run_does_not_execute_or_write_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts_dir, output_root, dash_dir = self._setup_workspace(root)
            called: list[str] = []

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = rdr.run_daily(
                    prompts_dir=prompts_dir,
                    output_root=output_root,
                    dashboard_reports_dir=dash_dir,
                    run_date="2026-11-14",
                    dry_run=True,
                    runner=lambda txt, d, t: called.append(txt) or "",
                )
            self.assertEqual(rc, 0)
            self.assertEqual(called, [])
            plan = json.loads(buf.getvalue())
            self.assertTrue(plan["dry_run"])
            self.assertEqual(len(plan["prompts"]), 2)
            self.assertFalse((output_root / "2026-11-14").exists())
            self.assertFalse(dash_dir.exists())

    def test_idempotency_partial_failure_and_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts_dir, output_root, dash_dir = self._setup_workspace(root)
            calls: list[str] = []
            fail_beta = True

            def fake_runner(prompt_text: str, run_date: str, timeout_sec: int) -> str:
                calls.append(prompt_text.splitlines()[0])
                if "Beta" in prompt_text and fail_beta:
                    raise RuntimeError("simulated timeout")
                return f"## 要約\n\n- 結果 ({run_date})\n\n| key | val |\n| --- | --- |\n| ok | yes |\n"

            # First run: alpha succeeds, beta fails -> exit code 1
            rc1 = rdr.run_daily(
                prompts_dir=prompts_dir,
                output_root=output_root,
                dashboard_reports_dir=dash_dir,
                run_date="2026-11-14",
                runner=fake_runner,
            )
            self.assertEqual(rc1, 1)
            self.assertEqual(len(calls), 2)

            alpha_md = output_root / "2026-11-14" / "20261114-01-alpha.md"
            alpha_html = output_root / "2026-11-14" / "20261114-01-alpha.html"
            alpha_dash = dash_dir / "20261114-01-alpha.html"
            self.assertTrue(alpha_md.is_file())
            self.assertTrue(alpha_html.is_file())
            self.assertTrue(alpha_dash.is_file())
            self.assertEqual(alpha_html.read_text(encoding="utf-8"), alpha_dash.read_text(encoding="utf-8"))

            state = json.loads((output_root / "2026-11-14" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["prompts"]["01-alpha"]["status"], "completed")
            self.assertEqual(state["prompts"]["02-beta"]["status"], "failed")

            # Second run without --force: skips alpha, retries only beta -> exit code 0
            calls.clear()
            fail_beta = False
            rc2 = rdr.run_daily(
                prompts_dir=prompts_dir,
                output_root=output_root,
                dashboard_reports_dir=dash_dir,
                run_date="2026-11-14",
                runner=fake_runner,
            )
            self.assertEqual(rc2, 0)
            self.assertEqual(calls, ["# Beta 調査"])

            state2 = json.loads((output_root / "2026-11-14" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state2["prompts"]["01-alpha"]["status"], "completed")
            self.assertEqual(state2["prompts"]["02-beta"]["status"], "completed")
            self.assertTrue((dash_dir / "20261114-02-beta.html").is_file())

            # Third run without --force: skips both
            calls.clear()
            rc3 = rdr.run_daily(
                prompts_dir=prompts_dir,
                output_root=output_root,
                dashboard_reports_dir=dash_dir,
                run_date="2026-11-14",
                runner=fake_runner,
            )
            self.assertEqual(rc3, 0)
            self.assertEqual(calls, [])

            # Fourth run with --force: re-runs both
            calls.clear()
            rc4 = rdr.run_daily(
                prompts_dir=prompts_dir,
                output_root=output_root,
                dashboard_reports_dir=dash_dir,
                run_date="2026-11-14",
                force=True,
                runner=fake_runner,
            )
            self.assertEqual(rc4, 0)
            self.assertEqual(len(calls), 2)

    def test_cli_main_with_mock_agy_in_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts_dir, output_root, dash_root = self._setup_workspace(root)
            fake_home = root / "home"
            cli_dir = fake_home / ".gemini" / "antigravity-cli"
            cli_dir.mkdir(parents=True)
            (cli_dir / "antigravity-oauth-token").write_text("oauth-tok", encoding="utf-8")
            (cli_dir / "bin").mkdir()
            (cli_dir / "builtin").mkdir()

            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_agy = bin_dir / "agy"
            fake_agy.write_text(
                """#!/usr/bin/env python3
import json
print(json.dumps({"status": "SUCCESS", "response": "## CLIテスト\\n\\n- 完了"}))
""",
                encoding="utf-8",
            )
            fake_agy.chmod(0o755)

            env = os.environ.copy()
            env["HOME"] = str(fake_home)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["DASHBOARD_DATA"] = str(dash_root)
            env["DAILY_RESEARCH_PROMPTS_DIR"] = str(prompts_dir)
            env["DAILY_RESEARCH_OUTPUT_DIR"] = str(output_root)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(DAILY_RESEARCH_DIR / "run_daily_research.py"),
                    "--only",
                    "01",
                    "--date",
                    "2026-11-14",
                    "--timeout",
                    "10",
                ],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            report = dash_root / "reports" / "daily-research" / "20261114-01-alpha.html"
            self.assertTrue(report.is_file())
            self.assertIn("CLIテスト", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
