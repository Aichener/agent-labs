#!/usr/bin/env python3
"""推送前自检：敏感表述扫描 + 禁用文件检查 + 单元测试 + 质量门禁。

用法：
    python tools/preflight.py            # 只检查
    python tools/preflight.py --fast     # 跳过测试（只做静态扫描）

退出码：0 通过；1 有问题（阻止推送）。

设计说明：
  - 敏感词表按「字」拆分存储，避免脚本自身被自己扫到；
  - 扫描跳过本文件与 .git/、日志、缓存；
  - 测试与门禁都用各项目自己的入口，不重复实现口径。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()

# 拆分存储，防止自检脚本自己被扫到
_BANNED_CHARS = [
    ("涉", "密"), ("定", "密"), ("密", "级"), ("机", "密"),
    ("军", "工"), ("部", "队"), ("研", "究", "所"),
    ("昇", "腾"), ("鲲", "鹏"), ("国", "防"),
    ("验", "收", "报", "告"), ("面", "试"), ("作", "品", "集"),
    ("h", "a", "r", "n", "e", "s", "s"),
    ("上", "线", "差", "距"), ("真", "实", "客", "户"),
]
BANNED = ["".join(parts) for parts in _BANNED_CHARS]

FORBIDDEN_NAMES = {"PUSH.md", "push.ps1", ".env", "tasks.db", "tasks.db-wal", "tasks.db-shm"}
FORBIDDEN_DIRS = {"logs", "__pycache__", ".venv", ".git"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
SCAN_SUFFIXES = {".py", ".md", ".json", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".ini"}
PROJECTS = ["contract-agent", "doc-classify-agent"]


def git_files() -> list[Path] | None:
    """要检查的文件：git 跟踪的 + 未跟踪但没被 ignore 的。

    只扫文件系统会把 logs/、data/tasks.db 这类忽略文件也算进来，误报一堆。
    """
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             cwd=str(ROOT), capture_output=True, timeout=30)
        text = out.stdout.decode("utf-8", "replace")
        rels = [r for r in text.split(chr(0)) if r]
        if out.returncode != 0:
            return None
        return [ROOT / r for r in rels if (ROOT / r).is_file()]
    except Exception:  # noqa: BLE001 —— 不是 git 仓库时退回文件系统扫描
        return None


def iter_files() -> list[Path]:
    files = git_files()
    if files is not None:
        return files
    found: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in FORBIDDEN_DIRS for part in rel.parts):
            continue
        found.append(path)
    return found


def check_sensitive(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        if path == SELF:
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES and path.name not in {"Dockerfile", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for word in BANNED:
            if word in text:
                line = text[: text.index(word)].count("\n") + 1
                problems.append(f"{path.relative_to(ROOT).as_posix()}:{line} 命中「{word}」")
    return problems


def check_files(files: list[Path]) -> list[str]:
    problems: list[str] = []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.name in FORBIDDEN_NAMES:
            problems.append(f"{rel} 不应提交（内部运维/环境文件）")
        if path.suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"{rel} 不应提交（编译产物）")
        if path.name in {"memory.json", "review_memory.json"}:
            problems.append(f"{rel} 不应提交（运行期记忆，含历史统计）")
    return problems


def run(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()


def check_projects(fast: bool) -> list[str]:
    problems: list[str] = []
    for name in PROJECTS:
        proj = ROOT / name
        if not proj.exists():
            problems.append(f"{name} 目录不存在")
            continue
        if fast:
            continue
        ok, out = run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], proj)
        tail = out.splitlines()[-1] if out.splitlines() else ""
        print(f"  [{ 'PASS' if ok else 'FAIL' }] {name} 单元测试  {tail}")
        if not ok:
            problems.append(f"{name} 单元测试未通过")

        gate = proj / "eval" / "gate.py"
        if gate.exists():
            ok, out = run([sys.executable, str(gate)], proj)
            print(f"  [{ 'PASS' if ok else 'FAIL' }] {name} 质量门禁")
            if not ok:
                problems.append(f"{name} 质量门禁未通过：{out.splitlines()[-1] if out.splitlines() else ''}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过测试与门禁，只做静态检查")
    args = ap.parse_args()

    files = iter_files()
    print(f"仓库：{ROOT}")
    print(f"文件：{len(files)}")

    problems = check_files(files)
    sensitive = check_sensitive(files)
    problems += sensitive

    print(f"  [{'PASS' if not sensitive else 'FAIL'}] 敏感表述扫描（{len(BANNED)} 个词）")
    if not args.fast:
        problems += check_projects(args.fast)

    print("-" * 56)
    if problems:
        log = ROOT / "logs" / "preflight_last.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(problems), encoding="utf-8")
        print(f"发现 {len(problems)} 个问题，已阻止推送（明细见 {log}）：")
        for p in problems:
            print(f"  [x] {p}")
        return 1
    print("自检通过，可以推送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
