"""CLI entrypoint regression tests."""

import runpy
import sys

import pytest


def test_python_m_evas_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evas", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("evas", run_name="__main__")

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "EVAS" in out
    assert "simulate" in out
