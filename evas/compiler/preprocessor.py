"""
preprocessor.py — Handle `include, `define, `default_transition directives
"""
import re
from pathlib import Path

VAMS_INCLUDE_DIR = Path(__file__).resolve().parent.parent / 'vams'


class PreprocessorError(Exception):
    pass


def preprocess(source: str, source_dir: str = '.', defines: dict = None,
               include_dirs: list = None) -> tuple:
    """
    Preprocess Verilog-A source: handle `include, `define, `default_transition.

    Returns (preprocessed_source, defines_dict, default_transition_value)
    """
    if defines is None:
        defines = {}
    if include_dirs is None:
        include_dirs = []

    include_dirs = [source_dir, str(VAMS_INCLUDE_DIR)] + include_dirs
    default_transition = None
    included_files = set()

    result = _preprocess_recursive(source, defines, include_dirs,
                                   included_files, default_transition)
    return result[0], defines, result[2]


def _preprocess_recursive(source, defines, include_dirs, included_files,
                          default_transition):
    lines = source.split('\n')
    output_lines = []
    condition_stack = []

    def is_active():
        return all(frame["active"] for frame in condition_stack)

    for line in lines:
        stripped = line.strip()

        m = re.match(r'`ifdef\s+(\w+)', stripped)
        if m:
            parent_active = is_active()
            cond = m.group(1) in defines
            active = parent_active and cond
            condition_stack.append(
                {
                    "parent_active": parent_active,
                    "active": active,
                    "branch_taken": active,
                    "seen_else": False,
                }
            )
            output_lines.append('')
            continue

        m = re.match(r'`ifndef\s+(\w+)', stripped)
        if m:
            parent_active = is_active()
            cond = m.group(1) not in defines
            active = parent_active and cond
            condition_stack.append(
                {
                    "parent_active": parent_active,
                    "active": active,
                    "branch_taken": active,
                    "seen_else": False,
                }
            )
            output_lines.append('')
            continue

        m = re.match(r'`elsif\s+(\w+)', stripped)
        if m:
            if not condition_stack:
                raise PreprocessorError("`elsif without matching `ifdef/`ifndef")
            frame = condition_stack[-1]
            if frame["seen_else"]:
                raise PreprocessorError("`elsif after `else")
            cond = m.group(1) in defines
            active = frame["parent_active"] and not frame["branch_taken"] and cond
            frame["active"] = active
            frame["branch_taken"] = frame["branch_taken"] or active
            output_lines.append('')
            continue

        if re.match(r'`else\b', stripped):
            if not condition_stack:
                raise PreprocessorError("`else without matching `ifdef/`ifndef")
            frame = condition_stack[-1]
            if frame["seen_else"]:
                raise PreprocessorError("duplicate `else in conditional block")
            frame["active"] = frame["parent_active"] and not frame["branch_taken"]
            frame["branch_taken"] = True
            frame["seen_else"] = True
            output_lines.append('')
            continue

        if re.match(r'`endif\b', stripped):
            if not condition_stack:
                raise PreprocessorError("`endif without matching `ifdef/`ifndef")
            condition_stack.pop()
            output_lines.append('')
            continue

        active = is_active()
        if not active:
            output_lines.append('')
            continue

        # `include "filename"
        m = re.match(r'`include\s+"([^"]+)"', stripped)
        if m:
            fname = m.group(1)
            content = _resolve_include(fname, include_dirs, included_files)
            if content is None:
                raise PreprocessorError(f"include file not found: {fname}")
            sub_result = _preprocess_recursive(content, defines,
                                               include_dirs, included_files,
                                               default_transition)
            output_lines.append(sub_result[0])
            if sub_result[2] is not None:
                default_transition = sub_result[2]
            continue

        # `define NAME value
        m = re.match(r'`define\s+(\w+)\s+(.*)', stripped)
        if m:
            name = m.group(1)
            value = m.group(2).strip()
            defines[name] = value
            continue

        # `undef NAME
        m = re.match(r'`undef\s+(\w+)', stripped)
        if m:
            defines.pop(m.group(1), None)
            continue

        # `default_transition value
        m = re.match(r'`default_transition\s+(\S+)', stripped)
        if m:
            default_transition = _parse_si_number(m.group(1))
            continue

        # Apply macro substitutions
        processed = line
        for name, value in defines.items():
            processed = processed.replace(f'`{name}', value)

        output_lines.append(processed)

    if condition_stack:
        raise PreprocessorError("unterminated conditional block: missing `endif")

    return '\n'.join(output_lines), defines, default_transition


def _resolve_include(filename, include_dirs, included_files):
    """Find and read an include file."""
    for d in include_dirs:
        path = Path(d) / filename
        if path.exists():
            rpath = str(path.resolve())
            if rpath in included_files:
                return ''  # Already included
            included_files.add(rpath)
            return path.read_text(encoding='utf-8', errors='replace')
    # Try without extension changes
    return None


def _parse_si_number(s):
    """Parse a number with optional SI suffix."""
    suffixes = {
        'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3, 'K': 1e3,
        'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18,
    }
    s = s.strip()
    if not s:
        return 0.0
    if s[-1] in suffixes:
        return float(s[:-1]) * suffixes[s[-1]]
    return float(s)
