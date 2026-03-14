# EVAS — Project Instructions for Claude

## Analyze scripts (analyze_*.py) conventions

- **Wall-clock time**: always measure simulator wall-clock time with `time.perf_counter()` around `evas_simulate()`. Display in plot titles as `wall clock: X.XXXX s` (4 decimal places, seconds).
- **Y-axis range**: set to `[-0.1·VDD, 1.2·VDD]` for digital waveform plots.
- **Output path**: `_DEFAULT_OUT` must resolve to `<repo-root>/output/<example>/`. From `evas/examples/<name>/`, that is `HERE.parent.parent.parent / 'output' / '<name>'`. Always wrap with `os.environ.get('EVAS_OUTPUT_DIR')` fallback for `evas run` compatibility.
- **Multiple plots**: prefer one plot per configuration/ratio/sweep-point rather than a combined multi-row figure, unless the user asks otherwise.
- **Plot title**: include signal-level info (period, ratio, etc.) and wall-clock time.

## Test conventions

- Unit tests live in `tests/test_engine.py` (engine primitives), `tests/test_compiler.py` (lexer + parser), `tests/test_examples.py` (functional end-to-end).
- Functional example tests use `validate_csv()` (or named validate function) from `evas/examples/<name>/validate_*.py`.
- No smoke tests — functional tests already simulate internally.
- Use `tmp_path` fixture for output isolation (parallel-safe).

## Docs

- Two Sphinx docs: `docs_zh/` (Chinese, default at `/zh/`) and `docs_en/` (English, at root).
- Custom CSS: `_static/custom.css` — Apple Developer Documentation aesthetic (SF Pro, system fonts, `#0071e3` blue, clean white/charcoal).
- busuanzi CDN: use `https://cdn.bootcdn.net/ajax/libs/busuanzi/2.3.0/bsz.pure.mini.js`.
- CI: `.github/workflows/docs.yml` builds on push to `main` and deploys to `evas.tokenzhang.com`.
