#!/usr/bin/env python3
"""Airender developer harness: one entry point for building, testing, checking and scaffolding.

Designed so that a human or an AI agent can iterate with a single command and get a short,
unambiguous answer (every step prints `ok` / `FAIL`, the exit code is non-zero on failure, and
`--json` prints a machine readable summary).

    python3 tools/air.py check              # format + layering + build + tests (the pre-push gate)
    python3 tools/air.py check --full       # + clang-tidy + ThreadSanitizer run
    python3 tools/air.py test [filter]      # build and run unit tests (substring filter)
    python3 tools/air.py shaders [filter]   # compile every HLSL shader with the build's dxc commands
    python3 tools/air.py run -- render --features pathtracer --frames 2 --report out/report.json
    python3 tools/air.py golden-update      # rewrite tests/golden/* after an intended change
    python3 tools/air.py new-feature bloom  # scaffold src/features/bloom from the example feature
    python3 tools/air.py format [--check] | tidy | layers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ["src", "apps", "tests"]
CPP_SUFFIXES = {".h", ".hpp", ".cpp"}

PRESETS = {
    "debug": {"dir": "build", "args": ["-DCMAKE_BUILD_TYPE=Debug"]},
    "release": {"dir": "build-release", "args": ["-DCMAKE_BUILD_TYPE=Release"]},
    "asan": {"dir": "build-asan", "args": ["-DCMAKE_BUILD_TYPE=Debug", "-DAIR_ENABLE_ASAN=ON"]},
    "tsan": {"dir": "build-tsan", "args": ["-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DAIR_ENABLE_TSAN=ON"]},
}

# Module layering: a module may include only the modules listed for it. Features may additionally
# include their own headers, never another feature's.
LAYERS = {
    "core": {"core"},
    "rhi": {"core", "rhi"},
    "rdg": {"core", "rhi", "rdg"},
    "renderer": {"core", "rhi", "rdg", "renderer"},
    "feature": {"core", "rhi", "rdg", "renderer"},
}

RESULTS: list[dict] = []


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def log(message: str) -> None:
    print(f"[air] {message}", flush=True)


def run_step(name: str, cmd: list[str], env: dict | None = None, cwd: Path = ROOT, tail: int = 60) -> bool:
    """Runs a command, prints `ok`/`FAIL` and on failure the last `tail` lines of its output."""
    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace")
    elapsed = time.monotonic() - start
    ok = proc.returncode == 0
    RESULTS.append({"step": name, "ok": ok, "seconds": round(elapsed, 2), "exitCode": proc.returncode})
    log(f"{name}: {'ok' if ok else 'FAIL'} ({elapsed:.1f}s)")
    if not ok:
        lines = proc.stdout.splitlines()
        print("\n".join(lines[-tail:]))
    return ok


def record(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"step": name, "ok": ok, "detail": detail})
    log(f"{name}: {'ok' if ok else 'FAIL'}{' - ' + detail if detail else ''}")
    return ok


def source_files(suffixes: set[str] = CPP_SUFFIXES) -> list[Path]:
    files = []
    for d in SOURCE_DIRS:
        files += [p for p in (ROOT / d).rglob("*") if p.suffix in suffixes and p.is_file()]
    return sorted(files)


def build_dir(preset: str) -> Path:
    return ROOT / PRESETS[preset]["dir"]


def exe(path: Path) -> Path:
    if os.name == "nt":
        for candidate in (path.with_suffix(".exe"), path.parent / "Debug" / (path.name + ".exe"),
                          path.parent / "Release" / (path.name + ".exe")):
            if candidate.exists():
                return candidate
    return path


def find_tool(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------
def configure(preset: str) -> bool:
    directory = build_dir(preset)
    if (directory / "CMakeCache.txt").exists():
        return True
    cmd = ["cmake", "-S", str(ROOT), "-B", str(directory), *PRESETS[preset]["args"]]
    if find_tool("ninja"):
        cmd += ["-G", "Ninja"]
    return run_step(f"configure[{preset}]", cmd)


def build(preset: str) -> bool:
    return configure(preset) and run_step(f"build[{preset}]", ["cmake", "--build", str(build_dir(preset)), "--parallel"])


def test(preset: str, test_filter: str | None, env: dict | None = None) -> bool:
    if not build(preset):
        return False
    binary = exe(build_dir(preset) / "tests" / "air_tests")
    cmd = [str(binary)] + ([test_filter] if test_filter else [])
    if preset == "tsan" and sys.platform.startswith("linux") and find_tool("setarch"):
        cmd = ["setarch", "-R", *cmd]  # TSAN needs ASLR off on newer kernels
    return run_step(f"test[{preset}]" + (f"({test_filter})" if test_filter else ""), cmd, env=env, tail=120)


HLSL_SUFFIXES = {".hlsl", ".hlsli"}
HLSL_ELSE_RE = re.compile(r"^(\s*)\}\s*(else\b.*?)\s*\{\s*$")
HLSL_OPEN_RE = re.compile(r"^(\s*)(\S.*?)\s*\{\s*$")


def hlsl_files() -> list[Path]:
    return sorted(p for d in ["src"] for p in (ROOT / d).rglob("*") if p.suffix in HLSL_SUFFIXES)


def hlsl_brace_lines(lines: list[str]) -> list[str]:
    """Allman braces for HLSL, which clang-format cannot parse: a block brace that ends a line of
    code moves onto its own line, `} else {` becomes three lines. Nothing else changes."""
    out = []
    for line in lines:
        match = HLSL_ELSE_RE.match(line)
        if match:
            out += [f"{match[1]}}}", f"{match[1]}{match[2]}", f"{match[1]}{{"]
            continue
        match = HLSL_OPEN_RE.match(line)
        if match and not match[2].startswith("//") and not match[2].endswith(("=", "(", ",")):
            out += [f"{match[1]}{match[2]}", f"{match[1]}{{"]
            continue
        out.append(line)
    return out


def format_hlsl(check: bool) -> bool:
    problems = []
    for path in hlsl_files():
        lines = path.read_text(encoding="utf-8").split("\n")
        fixed = hlsl_brace_lines(lines)
        if fixed == lines:
            continue
        if check:
            first = next(i for i, (a, b) in enumerate(zip(lines, fixed)) if a != b)
            problems.append(f"{path.relative_to(ROOT)}:{first + 1}: opening brace must be on its own line")
        else:
            path.write_text("\n".join(fixed), encoding="utf-8", newline="")
    if problems:
        print("\n".join(problems))
    return record("format hlsl" + (" --check" if check else ""), not problems, f"{len(problems)} files" if problems else "")


def format_sources(check: bool) -> bool:
    clang_format = find_tool("clang-format", "clang-format-18", "clang-format-17")
    if not clang_format:
        return record("format", False, "clang-format not found")
    files = [str(p) for p in source_files()]
    if check:
        ok = run_step("format --check", [clang_format, "--dry-run", "--Werror", *files], tail=40)
    else:
        ok = run_step("format", [clang_format, "-i", *files])
    return format_hlsl(check) and ok


def tidy(preset: str, paths: list[str]) -> bool:
    runner = find_tool("run-clang-tidy", "run-clang-tidy-18")
    if not runner:
        return record("tidy", False, "run-clang-tidy not found")
    if not build(preset):
        return False
    regex = "|".join(re.escape(p) for p in paths) if paths else "(src|apps|tests)/"
    return run_step("tidy", [runner, "-p", str(build_dir(preset)), "-quiet", regex], tail=80)


def module_of(path: Path) -> tuple[str, str | None]:
    rel = path.relative_to(ROOT).parts
    if rel[0] != "src":
        return "unrestricted", None
    if rel[1] == "features":
        return "feature", rel[2]
    return rel[1], None


def check_layers() -> bool:
    include_re = re.compile(r'^\s*#\s*include\s+"air/([a-z_]+)(?:/([a-z0-9_]+))?/')
    violations = []
    for path in source_files():
        module, feature = module_of(path)
        if module not in LAYERS:
            continue
        allowed = LAYERS[module]
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            match = include_re.match(line)
            if not match:
                continue
            target, sub = match.group(1), match.group(2)
            if target == "features":
                ok = module == "feature" and sub == feature
            else:
                ok = target in allowed
            if not ok:
                violations.append(f"{path.relative_to(ROOT)}:{number}: {module} must not include air/{target}"
                                  + (f"/{sub}" if sub else ""))
    for v in violations:
        print(v)
    return record("layers", not violations, f"{len(violations)} violation(s)" if violations else "")


def golden_update(preset: str) -> bool:
    ok = test(preset, "Golden", env={"AIR_UPDATE_GOLDEN": "1"})
    if ok and find_tool("git"):
        subprocess.run(["git", "--no-pager", "diff", "--stat", "--", "tests/golden"], cwd=ROOT)
    return ok


def run_aircli(preset: str, args: list[str]) -> bool:
    if not build(preset):
        return False
    binary = exe(build_dir(preset) / "apps" / "aircli" / "aircli")
    log(f"aircli {' '.join(args)}")
    return subprocess.run([str(binary), *args], cwd=ROOT).returncode == 0


def new_feature(name: str) -> bool:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        return record("new-feature", False, "name must be lower_snake_case")
    template = ROOT / "src" / "features" / "example"
    target = ROOT / "src" / "features" / name
    if target.exists():
        return record("new-feature", False, f"{target.relative_to(ROOT)} already exists")
    pascal = "".join(part.capitalize() for part in name.split("_"))
    replacements = [("ExampleFeature", f"{pascal}Feature"), ("Example", pascal), ("example", name),
                    ("EXAMPLE", name.upper()), ("AIR_EX_", f"AIR_{name.upper()}_")]

    def rename(text: str) -> str:
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    for src in template.rglob("*"):
        dst = target / rename(str(src.relative_to(template)))
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(rename(src.read_text(encoding="utf-8")), encoding="utf-8")
    record("new-feature", True, f"created {target.relative_to(ROOT)} (re-run cmake to pick it up)")
    log(f"next: edit {target.relative_to(ROOT)}/src/{pascal}Feature.cpp, then `python3 tools/air.py test {pascal}`")
    # The build directory globs features at configure time; force a reconfigure.
    for preset in PRESETS:
        cache = build_dir(preset) / "CMakeCache.txt"
        if cache.exists():
            subprocess.run(["cmake", str(build_dir(preset))], cwd=ROOT, stdout=subprocess.DEVNULL)
    return True


def compile_shaders(preset: str, name_filter: str | None) -> bool:
    """Compiles every shader of the build's manifest (written by CMake, see cmake/AirShaders.cmake)
    with exactly the build's command lines into a scratch directory: a fast HLSL check that needs no
    C++ build and leaves the build's shader outputs (and a running editor) alone."""
    if not configure(preset):
        return False
    manifest_path = build_dir(preset) / "shaders" / "manifest.json"
    if not manifest_path.exists():
        return record("shaders", False, f"no {manifest_path.relative_to(ROOT)} (configure found no dxc)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scratch = build_dir(preset) / "shader-check"
    scratch.mkdir(parents=True, exist_ok=True)
    compiled = failed = 0
    for shader in manifest["shaders"]:
        name = f"{shader['name']}/{Path(shader['output']).name}"
        if name_filter and name_filter not in name:
            continue
        compiled += 1
        output = scratch / name.replace("/", "_")
        command = [manifest["compiler"], *shader["args"], shader["source"], "-Fo", str(output)]
        if not run_step(f"shader {name}", command, tail=40):
            failed += 1
    return record("shaders", compiled > 0 and failed == 0, f"{compiled} compiled, {failed} failed")


def check(full: bool) -> bool:
    ok = format_sources(check=True)
    ok = check_layers() and ok
    ok = test("debug", None) and ok
    if full:
        ok = tidy("debug", []) and ok
        ok = test("tsan", None) and ok
    return ok


# ---------------------------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print a JSON summary of all steps at the end")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_preset(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--preset", choices=sorted(PRESETS), default="debug")
        return p

    with_preset(sub.add_parser("configure"))
    with_preset(sub.add_parser("build"))
    p = with_preset(sub.add_parser("test"))
    p.add_argument("filter", nargs="?")
    p = with_preset(sub.add_parser("shaders", help="compile every shader from the build manifest"))
    p.add_argument("filter", nargs="?")
    p = with_preset(sub.add_parser("run", help="run aircli; pass its arguments after --"))
    p.add_argument("args", nargs=argparse.REMAINDER)
    with_preset(sub.add_parser("golden-update"))
    p = sub.add_parser("format")
    p.add_argument("--check", action="store_true")
    p = with_preset(sub.add_parser("tidy"))
    p.add_argument("paths", nargs="*")
    sub.add_parser("layers")
    p = sub.add_parser("check")
    p.add_argument("--full", action="store_true", help="also run clang-tidy and the TSAN test run")
    p = sub.add_parser("new-feature")
    p.add_argument("name")

    argv = sys.argv[1:]
    want_json = "--json" in argv  # accepted anywhere on the command line
    args = parser.parse_args([a for a in argv if a != "--json"])
    args.json = args.json or want_json
    if args.command == "configure":
        ok = configure(args.preset)
    elif args.command == "build":
        ok = build(args.preset)
    elif args.command == "test":
        ok = test(args.preset, args.filter)
    elif args.command == "shaders":
        ok = compile_shaders(args.preset, args.filter)
    elif args.command == "run":
        ok = run_aircli(args.preset, [a for a in args.args if a != "--"])
    elif args.command == "golden-update":
        ok = golden_update(args.preset)
    elif args.command == "format":
        ok = format_sources(args.check)
    elif args.command == "tidy":
        ok = tidy(args.preset, args.paths)
    elif args.command == "layers":
        ok = check_layers()
    elif args.command == "check":
        ok = check(args.full)
    else:
        ok = new_feature(args.name)

    if args.json:
        print(json.dumps({"ok": ok, "steps": RESULTS}, indent=2))
    log("SUCCESS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
