"""EVAS command-line interface."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _get_examples_root() -> Path:
    """Return the bundled examples directory."""
    try:
        from importlib.resources import files
        p = Path(str(files("evas.examples")))
        if p.is_dir():
            return p
    except Exception:
        pass
    return Path(__file__).parent / "examples"


def _list_examples() -> list[str]:
    root = _get_examples_root()
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and not d.name.startswith(".")
    )


def _pick_scs(dst: Path, name: str, tb: str | None) -> Path | None:
    """Return the .scs file to simulate, or None on error."""
    if tb:
        scs_file = dst / tb
        if not scs_file.exists():
            print(f"Error: testbench '{tb}' not found in {dst}", file=sys.stderr)
            return None
        return scs_file
    preferred = dst / f"tb_{name}.scs"
    if preferred.exists():
        return preferred
    scs_files = sorted(dst.glob("*.scs"))
    if not scs_files:
        print(f"Error: no .scs testbench found in {dst}", file=sys.stderr)
        return None
    return scs_files[0]


def cmd_list(_args: argparse.Namespace) -> int:
    names = _list_examples()
    if not names:
        print("No bundled examples found.", file=sys.stderr)
        return 1
    print(f"Available examples ({len(names)}):")
    for name in names:
        print(f"  {name}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from evas.netlist.runner import evas_simulate
    previous_engine = os.environ.get("EVAS_ENGINE")
    if args.engine:
        os.environ["EVAS_ENGINE"] = args.engine
    try:
        ok = evas_simulate(args.input, log_path=args.log, output_dir=args.output)
    finally:
        if args.engine:
            if previous_engine is None:
                os.environ.pop("EVAS_ENGINE", None)
            else:
                os.environ["EVAS_ENGINE"] = previous_engine
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    from evas.netlist.runner import evas_simulate

    name = args.name
    examples_root = _get_examples_root()
    src = examples_root / name
    if not src.is_dir():
        available = _list_examples()
        print(f"Error: example '{name}' not found.", file=sys.stderr)
        print(f"Available: {', '.join(available)}", file=sys.stderr)
        return 1

    # Copy example files to <cwd>/evas-run/<name>/
    dst = Path.cwd() / "evas-run" / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("_"):
            shutil.copy2(f, dst / f.name)

    # Resolve cross-example ahdl_include paths (e.g. "../other/file.va")
    # and copy the referenced files so relative paths work from dst.
    import re
    _ahdl_re = re.compile(r'ahdl_include\s+"([^"]+)"')
    for scs in dst.glob("*.scs"):
        for m in _ahdl_re.finditer(scs.read_text(encoding="utf-8")):
            inc_path = Path(m.group(1))
            if inc_path.parts[0] == "..":
                resolved = (src / inc_path).resolve()
                target = (dst / inc_path).resolve()
                if resolved.exists() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(resolved, target)

    # Simulate directly. The packaged default engine is EVAS1/Python; callers
    # can request EVAS2 explicitly when the Rust backend has been built.
    output_dir = Path.cwd() / "output" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    scs_file = _pick_scs(dst, name, args.tb)
    if scs_file is None:
        return 1

    print(f"Running example '{name}': {scs_file.name} → {output_dir}")
    previous_engine = os.environ.get("EVAS_ENGINE")
    if args.engine:
        os.environ["EVAS_ENGINE"] = args.engine
    try:
        ok = evas_simulate(str(scs_file), output_dir=str(output_dir))
    finally:
        if args.engine:
            if previous_engine is None:
                os.environ.pop("EVAS_ENGINE", None)
            else:
                os.environ["EVAS_ENGINE"] = previous_engine
    if not ok:
        return 1

    print(f"Output: {output_dir}")

    # Run analyze scripts if present
    import importlib.util
    for analyze_script in sorted(src.glob("analyze_*.py")):
        spec = importlib.util.spec_from_file_location(f"_analyze_{analyze_script.stem}", str(analyze_script))
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            mod.analyze(output_dir)
        except Exception as e:
            print(f"Warning: {analyze_script.name} failed: {e}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="evas",
        description="EVAS — Event-driven Verilog-A Simulator",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # evas simulate
    p_sim = sub.add_parser("simulate", help="Simulate a Spectre .scs netlist")
    p_sim.add_argument("input", help=".scs netlist file")
    p_sim.add_argument("-o", "--output", default="./output",
                       help="Output directory (default: ./output)")
    p_sim.add_argument("-log", help="Log file path")
    p_sim.add_argument(
        "--engine",
        choices=["python", "evas2", "rust2"],
        help="Engine override. The default is python; evas2/rust2 requires the Rust backend.",
    )
    p_sim.set_defaults(func=cmd_simulate)

    # evas run
    p_run = sub.add_parser("run", help="Run a bundled example")
    p_run.add_argument("name", help="Example name (see 'evas list')")
    p_run.add_argument("--tb", metavar="FILE",
                       help="Testbench filename override (default: tb_<name>.scs)")
    p_run.add_argument(
        "--engine",
        choices=["python", "evas2", "rust2"],
        help="Engine override. The default is python; evas2/rust2 requires the Rust backend.",
    )
    p_run.set_defaults(func=cmd_run)

    # evas list
    p_list = sub.add_parser("list", help="List all available examples")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
