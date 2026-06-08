"""Integrity checks for the backend migration capability manifest."""

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "evas" / "examples"
MANIFEST = EXAMPLES / "backend_migration_capability_manifest.json"
TEST_EXAMPLES = ROOT / "tests" / "test_examples.py"


def _load_manifest():
    with MANIFEST.open(encoding="utf-8") as f:
        return json.load(f)


def _all_example_testbenches() -> set[str]:
    return {
        f"{path.parent.name}/{path.name}"
        for path in sorted(EXAMPLES.glob("*/tb_*.scs"))
    }


def _pytest_example_testbenches() -> set[str]:
    tree = ast.parse(TEST_EXAMPLES.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.endswith(".scs") and "/" in node.value:
                found.add(node.value)
    return found


def _load_validator(module_rel: str):
    path = EXAMPLES / module_rel
    spec = importlib.util.spec_from_file_location("_migration_manifest_validator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_lists_every_bundled_example_testbench():
    manifest = _load_manifest()
    cases = manifest["cases"]

    listed = [case["tb"] for case in cases]
    assert len(listed) == len(set(listed)), "duplicate testbench entries in manifest"
    assert set(listed) == _all_example_testbenches()

    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids in manifest"


def test_manifest_tracks_current_pytest_functional_coverage():
    manifest = _load_manifest()
    covered = {
        case["tb"]
        for case in manifest["cases"]
        if case["covered_by_current_pytest"]
    }

    assert covered == _pytest_example_testbenches()


def test_manifest_validation_entries_are_resolvable():
    manifest = _load_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["backend_contract"]["current_default_backend"] == "python_dict"

    for case in manifest["cases"]:
        tb_path = EXAMPLES / case["tb"]
        assert tb_path.exists(), case["tb"]
        assert case["group"] == tb_path.parent.name
        assert case["required_signals"][0] == "time"
        assert case["coverage"]

        validation = case["validation"]
        kind = validation["kind"]
        if kind == "validator":
            mod = _load_validator(validation["module"])
            for fn_name in validation["functions"]:
                assert hasattr(mod, fn_name), (case["id"], fn_name)
        elif kind == "schema_only":
            assert validation["reason"]
            assert not case["covered_by_current_pytest"]
        else:
            raise AssertionError(f"unknown validation kind: {kind}")
