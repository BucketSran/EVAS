"""Unit tests for evas.netlist — Spectre parser and runner helpers.

Covers:
  - _normalize_node_name: Cadence \\<N\\> → <N>
  - _extract_nodes: node name normalization in parsed lines
  - parse_spectre: full Virtuoso-exported netlist including bus nodes,
      PDK model includes, tran options, degenerate sources
  - ahdl_include path fallback: absolute path → filename in scs dir
  - _add_spectre_source: degenerate pulse (no period, val0==val1),
      degenerate sine (no freq, ampl=0)
"""
import csv
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

from evas.netlist.runner import (
    SpectreSource,
    _add_spectre_source,
    _apply_evas_profile,
    _configured_evas_engine,
    _parse_required_trace_signals,
    _trace_nodes_for_signals,
    _trace_output_signals_for_request,
    _write_csv,
    evas_simulate,
)
from evas.netlist.spectre_parser import (
    _extract_nodes,
    _normalize_node_name,
    _parse_suffix_number,
    parse_spectre,
)
from evas.simulator.engine import SimResult, Simulator

RUST_CORE = Path(__file__).resolve().parents[1] / "evas" / "rust_core"


def _build_rust_core_or_skip():
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available")
    subprocess.run(["cargo", "build", "--release"], cwd=RUST_CORE, check=True)


def test_configured_evas_engine_normalizes_rust_aliases(monkeypatch):
    monkeypatch.delenv("EVAS_ENGINE", raising=False)
    assert _configured_evas_engine({}) == "python"
    assert _configured_evas_engine({"evas_engine": "evas-rust"}) == "evas-rust"
    assert _configured_evas_engine({"evas_engine": "evas2"}) == "evas-rust"
    assert _configured_evas_engine({"evas_engine": "rust2"}) == "evas-rust"

    monkeypatch.setenv("EVAS_ENGINE", "evas-rust")
    assert _configured_evas_engine({}) == "evas-rust"
    monkeypatch.setenv("EVAS_ENGINE", "evas2")
    assert _configured_evas_engine({}) == "evas-rust"


def test_evas2_cross_zero_event_uses_python_event_semantics(tmp_path, monkeypatch):
    _build_rust_core_or_skip()
    va_file = tmp_path / "cross_zero_latched.va"
    va_file.write_text(textwrap.dedent("""\
        `include "disciplines.vams"

        module cross_zero_latched(bit, out);
            input bit;
            output out;
            electrical bit, out;
            parameter real vth = 0.45;
            real y;

            analog begin
                @(initial_step or cross(V(bit) - vth, 0)) begin
                    y = (V(bit) > vth) ? 1.0 : 0.0;
                end
                V(out) <+ transition(y, 0, 1p, 1p);
            end
        endmodule
    """))
    scs_file = tmp_path / "tb_cross_zero_latched.scs"
    scs_file.write_text(textwrap.dedent("""\
        simulator lang=spectre
        global 0

        ahdl_include "cross_zero_latched.va"

        Vbit (bit 0) vsource type=pwl wave=[0 0 1n 0 1.1n 0.9 2n 0.9]
        XDUT (bit out) cross_zero_latched

        tran tran stop=2n maxstep=10p
        save bit out
    """))

    monkeypatch.setenv("EVAS_ENGINE", "evas2")
    out_dir = tmp_path / "out"
    assert evas_simulate(str(scs_file), output_dir=str(out_dir))
    data = np.genfromtxt(out_dir / "tran.csv", delimiter=",", names=True)
    idx = int(np.argmin(np.abs(data["time"] - 1.5e-9)))
    assert data["out"][idx] == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# _normalize_node_name
# ===========================================================================

class TestNormalizeNodeName:

    def test_plain_name_unchanged(self):
        assert _normalize_node_name("VDD") == "VDD"

    def test_escaped_bus_subscript(self):
        # File contains DOUT\<9\> — Python string has literal backslashes
        assert _normalize_node_name("DOUT\\<9\\>") == "DOUT<9>"

    def test_escaped_multi_digit(self):
        assert _normalize_node_name("BUS\\<31\\>") == "BUS<31>"

    def test_only_opening_bracket(self):
        # Partially-escaped names should still normalise what's there
        assert _normalize_node_name("DOUT\\<9>") == "DOUT<9>"

    def test_no_brackets(self):
        assert _normalize_node_name("clk_i") == "clk_i"


class TestRequiredTraceHelpers:

    def test_required_trace_parses_env_style_signal_list(self, monkeypatch):
        monkeypatch.setenv("EVAS_REQUIRED_TRACE_SIGNALS", "time,V(OUT_P), vinp;vinn OUT_P")

        assert _parse_required_trace_signals({}) == ["OUT_P", "vinp", "vinn"]

    def test_required_trace_selects_actual_nodes_case_insensitively(self):
        nodes = _trace_nodes_for_signals(["out_p", "vinp"], {"OUT_P", "vinp", "extra"})

        assert nodes == ["OUT_P", "vinp"]

    def test_required_trace_output_uses_actual_signal_names(self):
        selected = _trace_output_signals_for_request(["out_p", "vinp"], {"OUT_P", "vinp", "extra"})

        assert selected == ["OUT_P", "vinp"]


# ===========================================================================
# _extract_nodes
# ===========================================================================

class TestExtractNodes:

    def test_simple_two_node(self):
        name, nodes, rem = _extract_nodes("V1 (VDD 0) vsource dc=0.9")
        assert name == "V1"
        assert nodes == ["VDD", "0"]
        assert rem == "vsource dc=0.9"

    def test_bus_nodes_normalized(self):
        line = r"I1 (VDD VSS DOUT\<9\> DOUT\<0\>) mymod k=1"
        name, nodes, rem = _extract_nodes(line)
        assert name == "I1"
        assert nodes == ["VDD", "VSS", "DOUT<9>", "DOUT<0>"]

    def test_rejects_two_names_before_node_list(self):
        with pytest.raises(ValueError, match="exactly one name"):
            _extract_nodes("ramp_gen dut (clk rst code_0) param=1")


# ===========================================================================
# parse_spectre — Virtuoso-exported netlist
# ===========================================================================

REAL_NETLIST = textwrap.dedent(r"""
    // Library name: 2026_TISAR
    // Cell name: tb_adc
    // View name: schematic
    simulator lang=spectre
    global 0
    include "/home/process/tsmc28n/crn28ull.scs" section=pre_simu
    include "/home/process/tsmc28n/crn28ull.scs" section=tt_mom
    V2 (clk_i 0) vsource type=pulse val0=0 val1=0
    V1 (VDD 0) vsource dc=900.0m type=dc
    V0 (VSS 0) vsource type=dc
    I3 (VDD VSS vin_i clk_i DOUT\<9\> DOUT\<8\> DOUT\<7\> DOUT\<6\> DOUT\<5\> \
            DOUT\<4\> DOUT\<3\> DOUT\<2\> DOUT\<1\> DOUT\<0\>) adc_10b vrefp=1 \
            vrefn=0 tedge=1e-11
    V3 (vin_i 0) vsource type=sine
    simulatorOptions options psfversion="1.4.0" reltol=1e-4 vabstol=1e-6 \
        iabstol=1e-12 temp=27 tnom=27 scalem=1.0 gmin=1e-12 rforce=1
    tran tran stop=1u errpreset=conservative write="spectre.ic" \
        writefinal="spectre.fc" savetime=[600] annotate=status
    finalTimeOP info what=oppoint where=rawfile
    modelParameter info what=models where=rawfile
    element info what=inst where=rawfile
    outputParameter info what=output where=rawfile
    designParamVals info what=parameters where=rawfile
    primitives info what=primitives where=rawfile
    subckts info what=subckts where=rawfile
    save DOUT\<9\> DOUT\<8\> DOUT\<7\> DOUT\<6\> DOUT\<5\> DOUT\<4\> DOUT\<3\> \
        DOUT\<2\> DOUT\<1\> DOUT\<0\> clk_i VDD VSS
    saveOptions options save=allpub
    ahdl_include "/home/zhangz/TSMC28N/2026_TISAR/adc_10b/veriloga/veriloga.va"
""")


class TestParseSpectreRealNetlist:

    @pytest.fixture
    def netlist(self, tmp_path):
        scs = tmp_path / "tb_adc.scs"
        scs.write_text(REAL_NETLIST)
        return parse_spectre(str(scs))

    def test_title_extracted(self, netlist):
        assert "2026_TISAR" in netlist.title or "tb_adc" in netlist.title

    def test_temp_from_simulator_options(self, netlist):
        assert netlist.temp == pytest.approx(27.0)

    def test_tran_stop(self, netlist):
        assert netlist.tran is not None
        assert netlist.tran.stop == pytest.approx(1e-6)

    def test_pdk_includes_stored_but_not_blocking(self, netlist):
        # include lines are stored; no error is raised
        assert len(netlist.includes) == 2

    def test_ahdl_include_path_stored(self, netlist):
        assert len(netlist.ahdl_includes) == 1
        assert "veriloga.va" in netlist.ahdl_includes[0].path

    def test_sources_parsed(self, netlist):
        names = {s.name for s in netlist.sources}
        assert names == {"V0", "V1", "V2", "V3"}

    def test_spectre_suffix_parser_keeps_m_and_M_distinct(self):
        assert _parse_suffix_number("900m") == pytest.approx(0.9)
        assert _parse_suffix_number("100M") == pytest.approx(100e6)
        assert _parse_suffix_number("100Meg") == pytest.approx(100e6)

    def test_spectre_star_comments_are_ignored(self, tmp_path):
        scs = tmp_path / "tb_star_comment.scs"
        scs.write_text(textwrap.dedent("""\
            * Divide ratio code = 5
            simulator lang=spectre
            global 0
            * Another SPICE-style comment with (parentheses)
            V1 (out 0) vsource dc=1 type=dc
            tran tran stop=1n
            save out
        """))
        parsed = parse_spectre(str(scs))
        assert [src.name for src in parsed.sources] == ["V1"]
        assert parsed.tran is not None

    def test_vdd_dc_voltage(self, netlist):
        v1 = next(s for s in netlist.sources if s.name == "V1")
        assert v1.source_type == "dc"
        assert v1.params.get("dc") == pytest.approx(0.9)

    def test_instance_parsed(self, netlist):
        assert len(netlist.instances) == 1
        i3 = netlist.instances[0]
        assert i3.name == "I3"
        assert i3.model_name == "adc_10b"

    def test_instance_params(self, netlist):
        i3 = netlist.instances[0]
        assert i3.params.get("vrefp") == pytest.approx(1.0)
        assert i3.params.get("vrefn") == pytest.approx(0.0)
        assert i3.params.get("tedge") == pytest.approx(1e-11)

    def test_bus_nodes_normalized(self, netlist):
        """DOUT\\<9\\> must arrive as DOUT<9> (no backslashes)."""
        i3 = netlist.instances[0]
        # First 4 nodes are scalar; next 10 are DOUT<9>..DOUT<0>
        dout_nodes = i3.nodes[4:]
        assert dout_nodes[0] == "DOUT<9>"
        assert dout_nodes[-1] == "DOUT<0>"
        # No backslashes anywhere
        for n in dout_nodes:
            assert "\\" not in n

    def test_bus_nodes_count(self, netlist):
        i3 = netlist.instances[0]
        # VDD VSS vin_i clk_i + 10 DOUT bits
        assert len(i3.nodes) == 14

    def test_save_signals_normalized(self, netlist):
        """save DOUT\\<9\\> ... must be stored as DOUT<9> — no backslashes."""
        assert len(netlist.save_signals) == 13  # 10 DOUT + clk_i + VDD + VSS
        for sig in netlist.save_signals:
            assert "\\" not in sig, f"Signal {sig!r} still contains backslash"
        assert "DOUT<9>" in netlist.save_signals
        assert "DOUT<0>" in netlist.save_signals
        assert "clk_i" in netlist.save_signals


# ===========================================================================
# ahdl_include path fallback
# ===========================================================================

class TestAhdlIncludePathFallback:

    VA_SRC = textwrap.dedent("""\
        `include "disciplines.vams"
        module dummy(out);
        output voltage out;
        analog begin
            V(out) <+ 0.0;
        end
        endmodule
    """)

    def test_fallback_to_filename_in_scs_dir(self, tmp_path, capsys):
        """If ahdl_include has an absolute path that doesn't exist, EVAS should
        fall back to the bare filename in the same directory as the .scs file."""
        va_file = tmp_path / "veriloga.va"
        va_file.write_text(self.VA_SRC)

        scs_content = textwrap.dedent("""\
            `include "disciplines.vams"
            V1 (out 0) vsource dc=0 type=dc
            I1 (out) dummy
            tran tran stop=10n
            ahdl_include "/nonexistent/path/to/veriloga.va"
        """)
        scs_file = tmp_path / "tb.scs"
        scs_file.write_text(scs_content)

        from evas.netlist.runner import evas_simulate
        ok = evas_simulate(str(scs_file), output_dir=str(tmp_path / "out"))
        assert ok, "evas_simulate should succeed via filename fallback"

    def test_relative_path_still_works(self, tmp_path):
        """Relative ahdl_include path (bare filename) keeps working."""
        va_file = tmp_path / "veriloga.va"
        va_file.write_text(self.VA_SRC)

        scs_content = textwrap.dedent("""\
            V1 (out 0) vsource dc=0 type=dc
            I1 (out) dummy
            tran tran stop=10n
            ahdl_include "veriloga.va"
        """)
        scs_file = tmp_path / "tb.scs"
        scs_file.write_text(scs_content)

        from evas.netlist.runner import evas_simulate
        ok = evas_simulate(str(scs_file), output_dir=str(tmp_path / "out"))
        assert ok

    def test_reserved_identifier_uses_spectre_vacomp_diagnostic(self, tmp_path):
        va_file = tmp_path / "sin_port.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module sin_port(sin, out);
                input sin;
                output out;
                electrical sin, out;

                analog begin
                    V(out) <+ V(sin);
                end
            endmodule
        """))

        scs_file = tmp_path / "tb_sin_port.scs"
        scs_file.write_text(textwrap.dedent("""\
            simulator lang=spectre
            global 0
            ahdl_include "sin_port.va"

            Vsin (vin 0) vsource type=dc dc=0.2
            XDUT (vin out) sin_port

            tran tran stop=1n maxstep=100p
            save vin out
        """))
        log_path = tmp_path / "evas.log"

        ok = evas_simulate(
            str(scs_file),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )

        assert not ok
        log = log_path.read_text(encoding="utf-8")
        assert "WARNING: ahdl_include resolved" not in log
        assert 'ERROR (VACOMP-2174): "module sin_port(sin<<--? , out);"' in log
        assert 'line 3: Identifier "sin" is a reserved name for a built-in function.' in log
        assert (
            "Use an identifier that is not a reserved name for a built-in function."
            in log
        )

    def test_user_defined_function_call_allowed_by_netlist_runner(self, tmp_path):
        va_file = tmp_path / "fn_clamp.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module fn_clamp(in, out);
                input in;
                output out;
                electrical in, out;
                real out_v;

                function real clamp01;
                    input real x;
                    begin
                        if (x < 0.1) clamp01 = 0.1;
                        else if (x > 0.8) clamp01 = 0.8;
                        else clamp01 = x;
                    end
                endfunction

                analog begin
                    out_v = clamp01(V(in));
                    V(out) <+ out_v;
                end
            endmodule
        """))

        scs_file = tmp_path / "tb_fn_clamp.scs"
        scs_file.write_text(textwrap.dedent("""\
            simulator lang=spectre
            global 0
            ahdl_include "fn_clamp.va"

            Vin (in 0) vsource type=pwl wave=[ 0n 0.05 5n 0.05 6n 0.9 10n 0.9 ]
            XDUT (in out) fn_clamp

            tran tran stop=10n maxstep=100p
            save in out
        """))

        out_dir = tmp_path / "out"
        assert evas_simulate(str(scs_file), output_dir=str(out_dir))

        with (out_dir / "tran.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows
        low_row = min(rows, key=lambda row: abs(float(row["time"]) - 2e-9))
        high_row = min(rows, key=lambda row: abs(float(row["time"]) - 8e-9))
        assert float(low_row["out"]) == pytest.approx(0.1, abs=1e-6)
        assert float(high_row["out"]) == pytest.approx(0.8, abs=1e-6)


# ===========================================================================
# EVAS/Spectre startup conformance
# ===========================================================================

class TestInitialTimerTransitionConformance:

    VA_SRC = textwrap.dedent("""\
        `include "disciplines.vams"
        module timer0_transition_probe(vctrl, phase);
        input vctrl;
        output phase;
        electrical vctrl, phase;
        real ph;
        parameter real tr = 200p;

        analog begin
            @(initial_step) ph = 0.0;
            @(timer(0, 1n)) ph = ph + 0.03 + 0.09 * V(vctrl);
            V(phase) <+ transition(ph, 0, tr, tr);
        end
        endmodule
    """)

    SCS_SRC = textwrap.dedent("""\
        simulator lang=spectre
        global 0
        Vctrl (vctrl 0) vsource type=dc dc=0.1
        I0 (vctrl phase) timer0_transition_probe
        tran tran stop=2n maxstep=500p
        save vctrl phase
        ahdl_include "timer0_transition_probe.va"
    """)

    def test_timer_zero_sets_initial_transition_target(self, tmp_path):
        va_file = tmp_path / "timer0_transition_probe.va"
        va_file.write_text(self.VA_SRC)
        scs_file = tmp_path / "tb_timer0_transition_probe.scs"
        scs_file.write_text(self.SCS_SRC)

        from evas.netlist.runner import evas_simulate
        ok = evas_simulate(str(scs_file), output_dir=str(tmp_path / "out"))
        assert ok

        csv_file = tmp_path / "out" / "tran.csv"
        rows = list(csv.DictReader(csv_file.open()))
        points = [(float(row["time"]), float(row["phase"])) for row in rows]

        # Spectre treats the timer(0, ...) event as part of the initial solve,
        # so the first saved point is the settled target, not a ramp from 0.
        assert points[0][0] == pytest.approx(0.0)
        assert points[0][1] == pytest.approx(0.039, abs=1e-12)

        before_second_timer = [v for t, v in points if t < 1e-9 - 1e-18]
        assert before_second_timer
        assert all(v == pytest.approx(0.039, abs=1e-12) for v in before_second_timer)


# ===========================================================================
# _add_spectre_source — degenerate cases
# ===========================================================================

def _make_pulse(name, val0, val1, period=0.0, **extra):
    params = {"type": "pulse", "val0": val0, "val1": val1, "period": period}
    params.update(extra)
    return SpectreSource(name=name, node_pos="clk", node_neg="0",
                         source_type="pulse", params=params)


def _make_sine(name, freq=0.0, ampl=0.0, sinedc=0.0, **extra):
    params = {"type": "sine", "freq": freq, "ampl": ampl, "sinedc": sinedc}
    params.update(extra)
    return SpectreSource(name=name, node_pos="vin", node_neg="0",
                         source_type="sine", params=params)


class TestAddSpectreSourceDegenerateCases:

    def _sim(self):
        return Simulator()

    def test_constant_pulse_warns_and_becomes_dc(self):
        """val0=0, val1=0 -> DC 0V + warning."""
        src = _make_pulse("V2", val0=0.0, val1=0.0, period=0.0)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert len(warns) == 1
        assert "val0 == val1" in warns[0]
        # Source should be registered
        assert any(s.node == "clk" for s in sim.sources)

    def test_pulse_val0_eq_val1_warns_and_becomes_dc(self):
        """val0==val1 regardless of period → DC + warning."""
        src = _make_pulse("V2", val0=1.8, val1=1.8, period=10e-9)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert len(warns) == 1
        assert "val0 == val1" in warns[0]

    def test_pulse_missing_period_but_different_vals_warns(self):
        """val0!=val1 but period=0 -> Spectre-style one-shot + warning."""
        src = _make_pulse("Vclk", val0=0.0, val1=1.8, period=0.0,
                          delay=2e-9, rise=50e-12)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert len(warns) == 1
        assert "period not set" in warns[0]
        waveform = sim.sources[-1].waveform
        assert waveform(1.9e-9) == pytest.approx(0.0)
        assert waveform(2.025e-9) == pytest.approx(0.9)
        assert waveform(2.05e-9) == pytest.approx(1.8)
        assert waveform(20e-9) == pytest.approx(1.8)

    def test_nonperiodic_pulse_width_falls_once(self):
        """A no-period pulse with width falls once and remains at val0."""
        src = _make_pulse("Vrst", val0=0.0, val1=0.9, period=0.0,
                          delay=1e-9, rise=50e-12, fall=50e-12,
                          width=2e-9)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert len(warns) == 1
        waveform = sim.sources[-1].waveform
        assert waveform(0.5e-9) == pytest.approx(0.0)
        assert waveform(1.05e-9) == pytest.approx(0.9)
        assert waveform(3.05e-9) == pytest.approx(0.9)
        assert waveform(3.1e-9) == pytest.approx(0.0)
        assert waveform(10e-9) == pytest.approx(0.0)

    def test_pulse_valid_no_warnings(self):
        """Well-formed pulse → no warnings."""
        src = _make_pulse("Vclk", val0=0.0, val1=1.8, period=10e-9,
                          rise=50e-12, fall=50e-12)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert warns == []

    def test_pulse_width_matches_spectre_plateau_semantics(self):
        """Spectre pulse width is the high plateau after the rise ramp."""
        src = _make_pulse(
            "Vclk", val0=0.0, val1=0.9, period=1e-9,
            delay=100e-12, rise=20e-12, fall=20e-12, width=500e-12,
        )
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")

        assert warns == []
        waveform = sim.sources[-1].waveform
        assert waveform(110e-12) == pytest.approx(0.45)
        assert waveform(120e-12) == pytest.approx(0.9)
        assert waveform(620e-12) == pytest.approx(0.9)
        assert waveform(630e-12) == pytest.approx(0.45)
        assert waveform(640e-12) == pytest.approx(0.0)

    def test_sine_no_freq_warns_and_becomes_dc(self):
        """sine with freq=0 → DC + warning."""
        src = _make_sine("V3", freq=0.0, ampl=0.5, sinedc=0.45)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert len(warns) == 1
        assert "freq not set" in warns[0]

    def test_sine_ampl_zero_warns_and_becomes_dc(self):
        """sine with ampl=0 → DC + warning."""
        src = _make_sine("V3", freq=1e6, ampl=0.0, sinedc=0.45)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert len(warns) == 1
        assert "amplitude=0" in warns[0]

    def test_sine_valid_no_warnings(self):
        """Well-formed sine → no warnings."""
        src = _make_sine("Vin", freq=100e6, ampl=0.4, sinedc=0.45)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert warns == []

    def test_sine_spectre_dc_mag_aliases_no_warnings(self):
        """Spectre sine aliases dc/mag map to offset/amplitude."""
        src = SpectreSource(
            name="Vin",
            node_pos="vin",
            node_neg="0",
            source_type="sine",
            params={"type": "sine", "freq": 73e6, "dc": 0.45, "mag": 0.40},
        )
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert warns == []
        assert sim.sources[-1].waveform(0.0) == pytest.approx(0.45)
        assert sim.sources[-1].waveform(0.25 / 73e6) == pytest.approx(0.85)

    def test_sine_offset_amplitude_frequency_are_not_transient_aliases(self):
        """Spectre ignores these names for transient sine; EVAS must too."""
        src = SpectreSource(
            name="Vin",
            node_pos="vin",
            node_neg="0",
            source_type="sine",
            params={
                "type": "sine",
                "freq": 100e6,
                "amplitude": 0.2,
                "offset": 0.45,
            },
        )
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert warns == []
        assert sim.sources[-1].waveform(0.0) == pytest.approx(0.0)
        assert sim.sources[-1].waveform(0.25 / 100e6) == pytest.approx(1.0)

    def test_sine_vo_va_are_not_transient_aliases(self):
        src = SpectreSource(
            name="Vin",
            node_pos="vin",
            node_neg="0",
            source_type="sine",
            params={"type": "sine", "freq": 50e6, "vo": 0.45, "va": 0.15},
        )
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert warns == []
        assert sim.sources[-1].waveform(0.0) == pytest.approx(0.0)
        assert sim.sources[-1].waveform(0.25 / 50e6) == pytest.approx(1.0)

    def test_sine_noncanonical_names_are_ignored_but_uppercase_M_frequency_is_preserved(self, tmp_path):
        scs = tmp_path / "tb_sine_aliases.scs"
        scs.write_text(textwrap.dedent("""\
            Vvin (vin 0) vsource type=sine amplitude=0.1 offset=0.45 freq=100M
            tran tran stop=10n
            save vin
        """))

        netlist = parse_spectre(str(scs))
        sim = self._sim()
        warns = _add_spectre_source(sim, netlist.sources[0], "0")

        assert warns == []
        assert sim.sources[-1].waveform(0.0) == pytest.approx(0.0)
        assert sim.sources[-1].waveform(2.5e-9) == pytest.approx(1.0)

    def test_sine_canonical_sinedc_ampl_preserves_uppercase_M_frequency(self, tmp_path):
        scs = tmp_path / "tb_sine_canonical.scs"
        scs.write_text(textwrap.dedent("""\
            Vvin (vin 0) vsource type=sine sinedc=0.45 ampl=0.1 freq=100M
            tran tran stop=10n
            save vin
        """))

        netlist = parse_spectre(str(scs))
        sim = self._sim()
        warns = _add_spectre_source(sim, netlist.sources[0], "0")

        assert warns == []
        assert sim.sources[-1].waveform(0.0) == pytest.approx(0.45)
        assert sim.sources[-1].waveform(2.5e-9) == pytest.approx(0.55)

    def test_pwl_duplicate_times_are_rejected_like_spectre(self):
        src = SpectreSource(
            name="Vin",
            node_pos="vin",
            node_neg="0",
            source_type="pwl",
            params={"type": "pwl", "wave": [0.0, 0.0, 4e-9, 1.0, 4e-9, 0.0]},
        )
        sim = self._sim()
        with pytest.raises(ValueError, match="strictly increasing"):
            _add_spectre_source(sim, src, "0")


# ===========================================================================
# save / PWL regressions
# ===========================================================================

class TestNetlistRegressions:

    def test_save_bus_range_is_expanded(self, tmp_path):
        scs = tmp_path / "tb_bus_save.scs"
        scs.write_text(textwrap.dedent(r"""\
            save vin_i:3f clk_i DOUT\<9\>:0
        """))

        netlist = parse_spectre(str(scs))

        assert netlist.save_signals == [
            "vin_i", "clk_i",
            "DOUT<9>", "DOUT<8>", "DOUT<7>", "DOUT<6>", "DOUT<5>",
            "DOUT<4>", "DOUT<3>", "DOUT<2>", "DOUT<1>", "DOUT<0>",
        ]
        assert netlist.save_formats["vin_i"] == "3f"

    def test_implicit_multiline_pwl_wave_is_rejected(self, tmp_path):
        scs = tmp_path / "tb_multiline_pwl.scs"
        scs.write_text(textwrap.dedent("""\
            VIN (vin_i 0) vsource type=pwl wave=[
                0      0.0
                20.48u 1.0
            ]
        """))

        with pytest.raises(ValueError, match="multiline wave=\\[\\.\\.\\.\\] requires backslash"):
            parse_spectre(str(scs))


class TestCsvWriter:

    def test_write_csv_preserves_formats_and_rounds_integer_columns(self, tmp_path):
        result = SimResult(
            time=np.array([0.0, 1e-9]),
            signals={
                "vin": np.array([0.1254, 0.3756]),
                "code_code": np.array([0.49, 1.51]),
                "fine": np.array([1.23456789, 2.3456789]),
            },
            step_sizes=np.array([0.0, 1e-9]),
        )
        csv_path = tmp_path / "tran.csv"

        _write_csv(
            csv_path,
            result,
            ["vin", "code_code", "fine"],
            {"vin": "3f", "code_code": "d", "fine": "4e"},
        )

        assert csv_path.read_text(encoding="utf-8").splitlines() == [
            "time,vin,code_code,fine",
            "0.000000000000e+00,0.125,0,1.2346e+00",
            "1.000000000000e-09,0.376,2,2.3457e+00",
        ]

    def test_write_csv_python_fallback_matches_numpy_writer(self, tmp_path, monkeypatch):
        result = SimResult(
            time=np.array([0.0, 2e-9]),
            signals={"out": np.array([0.1, 0.2]), "state_code": np.array([1.4, 2.6])},
            step_sizes=np.array([0.0, 2e-9]),
        )
        fast_path = tmp_path / "fast.csv"
        fallback_path = tmp_path / "fallback.csv"

        _write_csv(fast_path, result, ["out", "state_code"], {"state_code": "d"})
        monkeypatch.setenv("EVAS_CSV_WRITER", "python")
        _write_csv(fallback_path, result, ["out", "state_code"], {"state_code": "d"})

        assert fast_path.read_text(encoding="utf-8") == fallback_path.read_text(encoding="utf-8")


class TestPwlParserRegressions:

    def test_backslash_continued_pwl_wave_is_parsed(self, tmp_path):
        scs = tmp_path / "tb_continued_pwl.scs"
        scs.write_text(textwrap.dedent("""\
            VIN (vin_i 0) vsource type=pwl wave=[ \\
                0      0.0 \\
                20.48u 1.0 \\
            ]
        """))

        netlist = parse_spectre(str(scs))
        source = netlist.sources[0]

        assert source.source_type == "pwl"
        assert source.params["wave"] == pytest.approx([0.0, 0.0, 20.48e-6, 1.0])

    def test_comma_separated_pwl_wave_is_parsed(self, tmp_path):
        scs = tmp_path / "tb_comma_pwl.scs"
        scs.write_text(textwrap.dedent("""\
            VIN (vin_i 0) vsource type=pwl wave=[0n, 0.0, 10n, 1.0, 20n, 0.0]
        """))

        netlist = parse_spectre(str(scs))
        source = netlist.sources[0]

        assert source.source_type == "pwl"
        assert source.params["wave"] == pytest.approx([0.0, 0.0, 10e-9, 1.0, 20e-9, 0.0])

    def test_inline_arithmetic_in_pwl_wave_is_rejected_like_spectre(self, tmp_path):
        scs = tmp_path / "tb_expr_pwl.scs"
        scs.write_text(textwrap.dedent("""\
            VIN (vin_i 0) vsource type=pwl wave=[0n 0.0 10n+100p 1.0]
        """))

        with pytest.raises(ValueError, match="inline arithmetic inside wave"):
            parse_spectre(str(scs))

    def test_parenthesized_pulse_parameter_list_is_rejected_like_spectre(self, tmp_path):
        scs = tmp_path / "tb_bad_pulse.scs"
        scs.write_text(textwrap.dedent("""\
            V0 (b0 0) vsource type=pulse (0 0.9 5n 0.1n 0.1n 5n 10n)
            tran tran stop=80n
            save b0
        """))

        with pytest.raises(ValueError, match="parenthesized vsource parameter list"):
            parse_spectre(str(scs))

    def test_invalid_two_name_instance_syntax_is_rejected(self, tmp_path):
        scs = tmp_path / "tb_bad_instance.scs"
        scs.write_text(textwrap.dedent("""\
            ramp_gen dut (clk rst code_0)
            tran tran stop=1n
        """))

        with pytest.raises(ValueError, match="exactly one name"):
            parse_spectre(str(scs))

    def test_bare_multiline_instance_requires_continuation(self, tmp_path):
        scs = tmp_path / "tb_bare_multiline_instance.scs"
        scs.write_text(textwrap.dedent("""\
            XDUT (clk rst code_0
                  out_0 out_1) my_model
            tran tran stop=1n
        """))

        with pytest.raises(ValueError):
            parse_spectre(str(scs))


class TestIndexedMigrationHarness:
    @pytest.fixture(autouse=True)
    def _legacy_python_engine_for_instrumentation_tests(self, monkeypatch):
        monkeypatch.setenv("EVAS_ENGINE", "python")

    def test_evas_simulate_runs_indexed_parity_when_opted_in(self, tmp_path, monkeypatch):
        va = tmp_path / "pass_through.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module pass_through(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ V(vin);
                end
            endmodule
        """))
        scs = tmp_path / "tb_pass_through.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) pass_through
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "pass_through.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_INDEXED_PARITY", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_indexed_parity = true" in log
        assert "Indexed parity check:" in log
        assert "passed: checked_signals=2" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_logs_indexed_snapshot_profile_when_opted_in(self, tmp_path, monkeypatch):
        va = tmp_path / "pass_through.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module pass_through(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ V(vin);
                end
            endmodule
        """))
        scs = tmp_path / "tb_pass_through.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) pass_through
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "pass_through.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_INDEXED_SNAPSHOT_PROFILE", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_indexed_snapshot_profile = true" in log
        assert "dict_prev_snapshot_s" in log
        assert "indexed_prev_snapshot_s" in log
        assert "Indexed snapshot profile:" in log
        assert "max_abs_diff = 0.0" in log

    def test_evas_simulate_logs_indexed_arrays_when_opted_in(self, tmp_path, monkeypatch):
        va = tmp_path / "pass_through.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module pass_through(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ V(vin);
                end
            endmodule
        """))
        scs = tmp_path / "tb_pass_through.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) pass_through
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "pass_through.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_INDEXED_ARRAYS", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_indexed_arrays = true" in log
        assert "indexed_array_err_ratio_reads" in log
        assert "Indexed array profile:" in log
        assert "Indexed model IO plan:" in log
        assert "mapped_port_count = 2" in log
        assert "output_count = 1" in log
        assert "scalar_state_count = 0" in log
        assert "integer_state_count = 0" in log
        assert "state_array_count = 0" in log
        assert "state_array_slot_count = 0" in log
        assert "static_voltage_read_count = 1" in log
        assert "static_output_write_count = 1" in log
        assert "event_body_voltage_read_count = 0" in log
        assert "event_trigger_voltage_count = 0" in log
        assert "event_voltage_read_count = 0" in log
        assert "dynamic_branch_access_count = 0" in log
        assert "dynamic_voltage_read_count = 0" in log
        assert "dynamic_output_write_count = 0" in log
        assert "output_write_throughs =" in log
        assert "post_model_sync_repairs = 0" in log
        assert "Indexed voltage read probe:" in log
        assert "Indexed voltage array reads:" in log
        assert "fallbacks = 0" in log
        assert "reads =" in log
        assert "mismatches = 0" in log
        assert "missing_nodes = 0" in log
        assert "max_abs_diff = 0.0" in log

    def test_evas_simulate_logs_indexed_state_storage_when_opted_in(
        self,
        tmp_path,
        monkeypatch,
    ):
        va = tmp_path / "stateful.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module stateful(vout);
                output vout;
                electrical vout;
                real x = 0.0;
                integer code = 0;
                real accum[0:1];

                analog begin
                    x = x + 0.25;
                    code = code + 1;
                    accum[1] = x + code;
                    V(vout) <+ accum[1];
                end
            endmodule
        """))
        scs = tmp_path / "tb_stateful.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            I0 (vout) stateful
            tran tran stop=2n step=1n
            save vout:3f
            ahdl_include "stateful.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_INDEXED_STATE_STORAGE", "1")
        monkeypatch.setenv("EVAS_STATE_LOCAL_FASTPATH", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_indexed_state_storage = true" in log
        assert "evas_state_local_fastpath = true" in log
        assert "indexed_state_storage_enabled = 1" in log
        assert "indexed_state_storage_models = 1" in log
        assert "indexed_state_storage_scalar_slots = 2" in log
        assert "indexed_state_storage_integer_slots = 1" in log
        assert "indexed_state_storage_array_slots = 2" in log
        assert "indexed_state_scalar_reads_total =" in log
        assert "indexed_state_scalar_writes_total =" in log
        assert "indexed_state_array_reads_total =" in log
        assert "indexed_state_array_writes_total =" in log
        assert "indexed_state_array_oob_writes_total = 0" in log

    def test_evas_simulate_logs_static_branch_fastpath_when_opted_in(self, tmp_path, monkeypatch):
        va = tmp_path / "pass_through.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module pass_through(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ V(vin);
                end
            endmodule
        """))
        scs = tmp_path / "tb_pass_through.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) pass_through
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "pass_through.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_STATIC_BRANCH_FASTPATH", "1")
        monkeypatch.setenv("EVAS_INDEXED_ARRAYS", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_static_branch_fastpath = true" in log
        assert "evas_indexed_arrays = true" in log
        assert "static_branch_direct_array_models = 1" in log
        assert "static_branch_direct_array_read_nodes = 1" in log
        assert "static_branch_direct_array_write_nodes = 1" in log
        assert "static_branch_fastpath_codegen_models = 1" in log
        assert "static_branch_fastpath_static_read_nodes = 1" in log
        assert "static_branch_fastpath_static_write_nodes = 1" in log
        assert "static_branch_fastpath_fallbacks_total = 0" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_logs_model_eval_profile_when_opted_in(self, tmp_path, monkeypatch):
        va = tmp_path / "pass_through.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module pass_through(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ V(vin);
                end
            endmodule
        """))
        scs = tmp_path / "tb_pass_through.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) pass_through
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "pass_through.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))
        default_log = log_path.read_text(encoding="utf-8")
        assert "Model timing counters:" not in default_log

        monkeypatch.setenv("EVAS_PROFILE_MODEL_EVAL", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_profile_model_eval = true" in log
        assert "Model timing counters:" in log
        assert "evaluate_calls =" in log
        assert "evaluate_s =" in log
        assert "post_update_s =" in log
        assert "prepare_step_s =" in log

    def test_evas_simulate_logs_rust_static_eval_when_opted_in(self, tmp_path, monkeypatch):
        _build_rust_core_or_skip()
        va = tmp_path / "gain.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module gain(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ 2.0 * V(vin) + 0.125;
                end
            endmodule
        """))
        scs = tmp_path / "tb_gain.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) gain
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "gain.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_RUST_STATIC_EVAL", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_rust_static_eval = true" in log
        assert "evas_indexed_arrays = true" in log
        assert "rust_static_eval_available = 1" in log
        assert "rust_static_eval_candidate_models = 1" in log
        assert "rust_static_eval_models = 1" in log
        assert "rust_static_eval_ops = 1" in log
        assert "rust_static_eval_segments = 1" in log
        assert "rust_static_eval_max_segment_models = 1" in log
        assert "rust_static_eval_node_voltage_syncs =" in log
        assert "rust_static_eval_deferred_output_syncs =" in log
        assert "rust_static_eval_lifecycle_model_skips =" in log
        assert "rust_static_eval_runtime_param_ops = 0" in log
        assert "rust_static_eval_coeff_eval_fallbacks = 0" in log
        assert "indexed_array_dirty_validation_enabled = 1" in log
        assert "indexed_array_dirty_syncs =" in log
        assert "rust_static_eval_errors = 0" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_defaults_to_python_engine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EVAS_ENGINE", raising=False)
        scs = tmp_path / "tb_default_python_source.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            VDD (vdd 0) vsource type=dc dc=1.8
            tran tran stop=2n step=1n
            save vdd:3f
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_engine = evas-rust" not in log
        assert "evas_rust_full_model_fastpath = true" not in log
        assert "evas_rust_full_model_required = true" not in log
        assert "evas_rust_required = true" not in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_evas_rust_option_requires_rust_full_model(self, tmp_path):
        _build_rust_core_or_skip()
        scs = tmp_path / "tb_evas_rust_source.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            VDD (vdd 0) vsource type=dc dc=1.8
            simulatorOptions options evas_engine=evas-rust evas_skip_source_error_control=true
            tran tran stop=2n step=1n
            save vdd:3f
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_engine = evas-rust" in log
        assert "evas_rust_full_model_fastpath = true" in log
        assert "evas_rust_full_model_required = true" in log
        assert "evas_rust_required = true" in log
        assert "rust_sim_program_enabled = 1" in log
        assert "rust_sim_program_source_record_enabled = 1" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_evas2_alias_logs_evas_rust(self, tmp_path):
        _build_rust_core_or_skip()
        scs = tmp_path / "tb_evas2_alias_source.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            VDD (vdd 0) vsource type=dc dc=1.8
            simulatorOptions options evas_engine=evas2 evas_skip_source_error_control=true
            tran tran stop=2n step=1n
            save vdd:3f
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_engine = evas-rust" in log
        assert "evas_engine = evas2" not in log
        assert "evas_rust_full_model_required = true" in log
        assert "rust_sim_program_enabled = 1" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_logs_rust_transition_shadow_when_opted_in(
        self,
        tmp_path,
        monkeypatch,
    ):
        _build_rust_core_or_skip()
        va = tmp_path / "trans_target_shadow.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module trans_target_shadow(inp, out);
                input inp;
                output out;
                electrical inp, out;
                integer q = 0;

                analog begin
                    q = V(inp) > 0.45 ? 1 : 0;
                    V(out) <+ transition(q ? 1.0 : 0.0, 0.0, 1n, 2n);
                end
            endmodule
        """))
        scs = tmp_path / "tb_trans_target_shadow.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (inp 0) vsource type=pulse val0=0 val1=1 period=2n width=1n rise=1p fall=1p
            I0 (inp out) trans_target_shadow
            tran tran stop=3n step=1n
            save inp:3f out:3f
            ahdl_include "trans_target_shadow.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_RUST_TRANSITION_SHADOW", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_rust_transition_shadow = true" in log
        assert "evas_indexed_arrays = true" in log
        assert "rust_transition_shadow_requested = 1" in log
        assert "rust_transition_shadow_available = 1" in log
        assert "rust_transition_shadow_candidate_models = 1" in log
        assert "rust_transition_shadow_models = 1" in log
        assert "rust_transition_shadow_static_ops = 1" in log
        assert "rust_transition_shadow_target_ops = 1" in log
        assert "rust_transition_shadow_segments = 1" in log
        assert "rust_transition_shadow_mismatches = 0" in log
        assert "rust_transition_shadow_errors = 0" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_logs_rust_event_due_shadow_when_opted_in(
        self,
        tmp_path,
        monkeypatch,
    ):
        _build_rust_core_or_skip()
        va = tmp_path / "event_due_shadow.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module event_due_shadow(inp, out);
                input inp;
                output out;
                electrical inp, out;
                integer count = 0;

                analog begin
                    @(cross(V(inp) - 0.5, +1)) count = count + 1;
                    @(above(V(inp) - 0.5)) count = count + 1;
                    @(timer(0, 1n)) count = count + 1;
                    @(timer(2n)) count = count + 1;
                    V(out) <+ count;
                end
            endmodule
        """))
        scs = tmp_path / "tb_event_due_shadow.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (inp 0) vsource type=pulse val0=0 val1=1 period=2n width=1n rise=1p fall=1p
            I0 (inp out) event_due_shadow
            tran tran stop=3n step=1n
            save inp:3f out:3f
            ahdl_include "event_due_shadow.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_RUST_EVENT_DUE_SHADOW", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_rust_event_due_shadow = true" in log
        assert "evas_indexed_arrays = true" not in log
        assert "rust_event_due_shadow_requested = 1" in log
        assert "rust_event_due_shadow_available = 1" in log
        assert "rust_event_due_shadow_enabled = 1" in log
        assert "rust_event_due_shadow_cross_checks_total =" in log
        assert "rust_event_due_shadow_above_checks_total =" in log
        assert "rust_event_due_shadow_timer_periodic_checks_total =" in log
        assert "rust_event_due_shadow_timer_absolute_checks_total =" in log
        assert "rust_event_due_shadow_mismatches_total = 0" in log
        assert "rust_event_due_shadow_errors_total = 0" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_logs_event_trace_audit_when_opted_in(
        self,
        tmp_path,
        monkeypatch,
    ):
        va = tmp_path / "event_trace_audit.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module event_trace_audit(inp, out);
                input inp;
                output out;
                electrical inp, out;
                real x = 0.0;
                real acc[0:1];

                analog begin
                    @(initial_step) begin
                        x = 1.0;
                        acc[0] = x;
                    end
                    @(cross(V(inp) - 0.5, +1)) begin
                        x = x + 1.0;
                        acc[1] = x;
                    end
                    @(timer(2n)) x = x + 1.0;
                    @(final_step) x = x + 1.0;
                    V(out) <+ x;
                end
            endmodule
        """))
        scs = tmp_path / "tb_event_trace_audit.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (inp 0) vsource type=pulse val0=0 val1=1 period=2n width=1n rise=1p fall=1p
            I0 (inp out) event_trace_audit
            tran tran stop=3n step=1n
            save inp:3f out:3f
            ahdl_include "event_trace_audit.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_EVENT_TRACE_AUDIT", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_event_trace_audit = true" in log
        assert "event_trace_audit_requested = 1" in log
        assert "event_trace_audit_enabled = 1" in log
        assert "event_trace_audit_events_total =" in log
        assert "event_trace_audit_state_writes_total =" in log
        assert "event_trace_audit_array_writes_total =" in log
        assert "event_trace_audit_output_writes_total =" in log
        assert "event_trace_audit_records_dropped_total = 0" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_logs_rust_static_fast_sync_when_opted_in(
        self,
        tmp_path,
        monkeypatch,
    ):
        _build_rust_core_or_skip()
        va = tmp_path / "gain.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module gain(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ 2.0 * V(vin) + 0.125;
                end
            endmodule
        """))
        scs = tmp_path / "tb_gain.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) gain
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "gain.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_RUST_STATIC_FAST_SYNC", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_rust_static_eval = true" in log
        assert "evas_rust_static_fast_sync = true" in log
        assert "rust_static_fast_sync_requested = 1" in log
        assert "rust_static_fast_sync_enabled = 1" in log
        assert "rust_static_fast_sync_node_voltage_sync_skips =" in log
        assert "rust_static_fast_sync_validation_skips =" in log
        assert "indexed_array_dirty_validation_enabled = 0" in log
        assert "indexed_array_syncs = 0" in log
        assert "rust_static_eval_errors = 0" in log
        assert (out_dir / "tran.csv").exists()

    def test_evas_simulate_rust_static_eval_uses_instance_parameter_overrides(
        self,
        tmp_path,
        monkeypatch,
    ):
        _build_rust_core_or_skip()
        va = tmp_path / "gain_param.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module gain_param(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;
                parameter real gain = 2.0;
                parameter real offset = 0.125;

                analog begin
                    V(vout) <+ gain * V(vin) + offset;
                end
            endmodule
        """))
        scs = tmp_path / "tb_gain_param.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) gain_param gain=3 offset=250m
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "gain_param.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        monkeypatch.setenv("EVAS_RUST_STATIC_EVAL", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "rust_static_eval_runtime_param_ops = 1" in log
        assert "rust_static_eval_coeff_eval_fallbacks = 0" in log
        assert "rust_static_eval_models = 1" in log
        rows = list(csv.DictReader((out_dir / "tran.csv").open()))
        assert float(rows[-1]["vout"]) == pytest.approx(2.5)

    def test_evas_simulate_logs_model_io_profile_when_opted_in(self, tmp_path, monkeypatch):
        va = tmp_path / "pass_through.va"
        va.write_text(textwrap.dedent("""\
            `include "disciplines.vams"

            module pass_through(vin, vout);
                input vin;
                output vout;
                electrical vin, vout;

                analog begin
                    V(vout) <+ V(vin);
                end
            endmodule
        """))
        scs = tmp_path / "tb_pass_through.scs"
        scs.write_text(textwrap.dedent("""\
            simulator lang=spectre
            V0 (vin 0) vsource type=dc dc=0.75
            I0 (vin vout) pass_through
            tran tran stop=2n step=1n
            save vin:3f vout:3f
            ahdl_include "pass_through.va"
        """))
        out_dir = tmp_path / "out"
        log_path = tmp_path / "evas.log"

        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))
        default_log = log_path.read_text(encoding="utf-8")
        assert "Model IO counters:" not in default_log

        monkeypatch.setenv("EVAS_PROFILE_MODEL_IO", "1")
        assert evas_simulate(str(scs), log_path=str(log_path), output_dir=str(out_dir))

        log = log_path.read_text(encoding="utf-8")
        assert "evas_profile_model_io = true" in log
        assert "Model IO counters:" in log
        assert "voltage_reads =" in log
        assert "voltage_read_local_nodes =" in log
        assert "voltage_read_external_nodes =" in log
        assert "output_writes =" in log
        assert "output_write_nodes =" in log


class TestEvasProfileMapping:

    def test_fast_profile_overrides_refinement(self):
        rf, rs, rt, applied = _apply_evas_profile("fast", 32, 16, 1e-4)
        assert (rf, rs, applied) == (8, 4, "fast")
        assert rt == pytest.approx(5e-3)

    def test_balanced_profile_keeps_defaults(self):
        rf, rs, rt, applied = _apply_evas_profile("balanced", 16, 8, 1e-3)
        assert (rf, rs, applied) == (16, 8, "balanced")
        assert rt == pytest.approx(1e-3)

    def test_precision_profile_caps_reltol(self):
        rf, rs, rt, applied = _apply_evas_profile("precision", 8, 4, 1e-2)
        assert (rf, rs, applied) == (32, 16, "precision")
        assert rt == pytest.approx(1e-4)

    def test_invalid_empty_pwl_reports_error_instead_of_crashing(self, tmp_path):
        scs = tmp_path / "tb_bad_pwl.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            VIN (vin_i 0) vsource type=pwl wave=[
            ]
            tran tran stop=1u
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        assert "ERROR: Failed to parse" in log_path.read_text()
        assert "multiline wave=[...] requires backslash" in log_path.read_text()


class TestSpectreCompatibilityPreflight:

    def test_initial_step_contribution_is_allowed(self, tmp_path):
        va_file = tmp_path / "initial_contribution.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module initial_contribution(out);
                output out;
                electrical out;
                real target;
                analog begin
                    @(initial_step) begin
                        target = 1.0;
                        V(out) <+ target;
                    end
                    V(out) <+ target;
                end
            endmodule
        """))
        scs = tmp_path / "tb_initial_contribution.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            I1 (out) initial_contribution
            tran tran stop=1n maxstep=100p
            ahdl_include "initial_contribution.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is True

    def test_event_body_contribution_fails(self, tmp_path):
        va_file = tmp_path / "event_contribution.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module event_contribution(clk, out);
                input clk;
                output out;
                electrical clk, out;
                analog begin
                    @(cross(V(clk) - 0.5, +1)) begin
                        V(out) <+ transition(1.0, 0, 1p, 1p);
                    end
                end
            endmodule
        """))
        scs = tmp_path / "tb_event_contribution.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            Vclk (clk 0) vsource type=pulse val0=0 val1=1 period=1n width=500p rise=1p fall=1p
            I1 (clk out) event_contribution
            tran tran stop=2n
            ahdl_include "event_contribution.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        assert "contribution statement" in log_path.read_text()

    def test_transition_in_runtime_case_block_fails(self, tmp_path):
        va_file = tmp_path / "bad_transition.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module bad_transition(inp, out);
                input inp;
                output out;
                electrical inp, out;
                integer i;
                analog begin
                    for (i = 0; i < 1; i = i + 1) begin
                        case (i)
                            0: V(out) <+ transition(V(inp), 0, 1n, 1n);
                        endcase
                    end
                end
            endmodule
        """))
        scs = tmp_path / "tb_bad_transition.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            Vin (inp 0) vsource dc=1 type=dc
            I1 (inp out) bad_transition
            tran tran stop=1n
            ahdl_include "bad_transition.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        assert "transition()" in log_path.read_text()

    def test_transition_in_if_branch_fails(self, tmp_path):
        va_file = tmp_path / "bad_transition_if.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module bad_transition_if(inp, out);
                input inp;
                output out;
                electrical inp, out;
                analog begin
                    if (V(inp) > 0.5)
                        V(out) <+ transition(1.0, 0, 1n, 1n);
                    else
                        V(out) <+ transition(0.0, 0, 1n, 1n);
                end
            endmodule
        """))
        scs = tmp_path / "tb_bad_transition_if.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            Vin (inp 0) vsource dc=1 type=dc
            I1 (inp out) bad_transition_if
            tran tran stop=1n
            ahdl_include "bad_transition_if.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        assert "transition()" in log_path.read_text()

    def test_standalone_wait_call_fails(self, tmp_path):
        va_file = tmp_path / "bad_wait.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module bad_wait(out);
                output out;
                electrical out;
                analog begin
                    wait(1n);
                    V(out) <+ 1.0;
                end
            endmodule
        """))
        scs = tmp_path / "tb_bad_wait.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            I1 (out) bad_wait
            tran tran stop=1n
            ahdl_include "bad_wait.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        assert "wait()" in log_path.read_text()

    def test_unknown_dollar_function_fails(self, tmp_path):
        va_file = tmp_path / "bad_itor.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module bad_itor(out);
                output out;
                electrical out;
                integer code;
                real val;
                analog begin
                    code = 3;
                    val = $itor(code);
                    V(out) <+ val;
                end
            endmodule
        """))
        scs = tmp_path / "tb_bad_itor.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            I1 (out) bad_itor
            tran tran stop=1n
            ahdl_include "bad_itor.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        assert "$itor()" in log_path.read_text()

    def test_tanh_math_function_is_supported(self, tmp_path):
        va_file = tmp_path / "tanh_probe.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module tanh_probe(inp, out);
                input inp;
                output out;
                electrical inp, out;
                analog begin
                    V(out) <+ tanh(V(inp));
                end
            endmodule
        """))
        scs = tmp_path / "tb_tanh_probe.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            Vin (inp 0) vsource dc=0.5 type=dc
            I1 (inp out) tanh_probe
            tran tran stop=1n
            ahdl_include "tanh_probe.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is True

    def test_supply_port_hard_drive_conflict_fails(self, tmp_path):
        va_file = tmp_path / "supply_driver.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module supply_driver(vdd, out);
                input vdd;
                output out;
                electrical vdd, out;
                analog begin
                    V(vdd) <+ 1.8;
                    V(out) <+ 0.0;
                end
            endmodule
        """))
        scs = tmp_path / "tb_supply_driver.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            VDD (vdd 0) vsource dc=1.8 type=dc
            I1 (vdd out) supply_driver
            tran tran stop=1n
            ahdl_include "supply_driver.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        log_text = log_path.read_text()
        assert "drives supply port" in log_text

    def test_parameter_port_name_collision_fails(self, tmp_path):
        va_file = tmp_path / "param_port_collision.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module param_port_collision(vdd, out);
                input vdd;
                output out;
                electrical vdd, out;
                parameter real vdd = 0.9;
                analog begin
                    V(out) <+ V(vdd);
                end
            endmodule
        """))
        scs = tmp_path / "tb_param_port_collision.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            VDD (vdd 0) vsource dc=0.9 type=dc
            I1 (vdd out) param_port_collision
            tran tran stop=1n
            ahdl_include "param_port_collision.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        log_text = log_path.read_text()
        assert "parameter name" in log_text
        assert "collides with module port" in log_text

    def test_case_distinct_parameter_and_port_names_are_allowed(self, tmp_path):
        va_file = tmp_path / "case_distinct_supply.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module case_distinct_supply(VDD, out);
                input VDD;
                output out;
                electrical VDD, out;
                parameter real vdd = 0.9;
                analog begin
                    V(out) <+ V(VDD) + 0.0 * vdd;
                end
            endmodule
        """))
        scs = tmp_path / "tb_case_distinct_supply.scs"
        scs.write_text(textwrap.dedent("""\
            VDD (vdd 0) vsource dc=0.9 type=dc
            I1 (vdd out) case_distinct_supply
            tran tran stop=1n
            ahdl_include "case_distinct_supply.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(str(scs), output_dir=str(tmp_path / "out"))
        assert ok

    def test_instance_terminal_count_mismatch_fails(self, tmp_path):
        va_file = tmp_path / "two_port.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module two_port(inp, out);
                input inp;
                output out;
                electrical inp, out;
                analog begin
                    V(out) <+ V(inp);
                end
            endmodule
        """))
        scs = tmp_path / "tb_short_instance.scs"
        log_path = tmp_path / "evas.log"
        scs.write_text(textwrap.dedent("""\
            Vin (inp 0) vsource dc=0.5 type=dc
            I1 (inp) two_port
            tran tran stop=1n
            ahdl_include "two_port.va"
            save out
        """))

        from evas.netlist.runner import evas_simulate

        ok = evas_simulate(
            str(scs),
            log_path=str(log_path),
            output_dir=str(tmp_path / "out"),
        )
        assert ok is False
        log_text = log_path.read_text()
        assert "terminal count mismatch" in log_text


class TestStochasticTransitionRampConformance:
    """Regression guard for the 06526fa stochastic-transition bypass.

    transition() targets derived from $rdist_normal must still produce a real
    ramp: the recorded row at the event timestamp keeps the pre-event value and
    the output midpoint lands ~tedge/2 after the tick (Spectre convention).
    The 06526fa bypass snapped the output instantaneously at the event row,
    which moved digital edges by ~half a record interval and broke certified
    EVAS/Spectre parity on the vaBench dither case (252 ps vs 1.19 ps).
    """

    VA_SRC = textwrap.dedent("""\
        `include "disciplines.vams"
        module stochastic_transition_probe(out);
        output out;
        electrical out;
        parameter real sigma = 0.01;
        parameter real tedge = 200p;
        real level;

        analog begin
            @(initial_step) level = 0.0;
            @(timer(1n)) level = 1.0 + sigma * $rdist_normal(0, 0, 1);
            V(out) <+ transition(level, 0, tedge, tedge);
        end
        endmodule
    """)

    SCS_SRC = textwrap.dedent("""\
        simulator lang=spectre
        global 0
        I0 (out) stochastic_transition_probe
        tran tran stop=3n maxstep=50p
        save out
        ahdl_include "stochastic_transition_probe.va"
    """)

    def test_rdist_transition_keeps_ramp_semantics(self, tmp_path):
        va_file = tmp_path / "stochastic_transition_probe.va"
        va_file.write_text(self.VA_SRC)
        scs_file = tmp_path / "tb_stochastic_transition_probe.scs"
        scs_file.write_text(self.SCS_SRC)

        from evas.netlist.runner import evas_simulate
        ok = evas_simulate(str(scs_file), output_dir=str(tmp_path / "out"))
        assert ok

        csv_file = tmp_path / "out" / "tran.csv"
        rows = [
            (float(r["time"]), float(r["out"]))
            for r in csv.DictReader(csv_file.open())
        ]
        level = max(v for _, v in rows)
        assert level > 0.9

        # Pre-event sample preserved: the last row at t <= 1n is still low.
        at_tick = [v for t, v in rows if t <= 1.0e-9 + 1.0e-15]
        assert at_tick, "no rows at or before the timer tick"
        assert at_tick[-1] < 0.1 * level, (
            f"pre-event sample lost at the tick row: {at_tick[-1]!r}"
        )

        # Ramp midpoint lands ~tedge/2 after the tick (linear interpolation).
        vth = 0.5 * level
        t_cross = None
        for (t0, v0), (t1, v1) in zip(rows, rows[1:]):
            if (v0 - vth) <= 0.0 < (v1 - vth):
                frac = (vth - v0) / (v1 - v0)
                t_cross = t0 + frac * (t1 - t0)
                break
        assert t_cross is not None, "no rising edge found"
        ideal = 1.0e-9 + 100.0e-12
        assert abs(t_cross - ideal) < 30.0e-12, (
            f"transition midpoint {t_cross} deviates from {ideal} by "
            f"{abs(t_cross - ideal) * 1e12:.1f} ps — stochastic transition"
            " bypass regression?"
        )


class TestCrossAcceptanceLawMode:
    """Opt-in Spectre-compatible cross event lateness (measured law).

    DOE 2026-06-12 (testspace/cross-lateness-doe-20260612): strict Spectre
    accepts a cross event late by Delta = 0.5 * reltol * |V_cross| / |slope|,
    independent of maxstep, errpreset and explicit time_tol. With
    evas_cross_acceptance_slack_factor=1.0 EVAS reproduces that law on
    pre-phase (source-driven, ground-referenced) crossings; validated against
    measured Spectre deltas to sub-fs. Default (factor unset) stays exact.
    """

    VA_SRC = textwrap.dedent("""\
        `include "disciplines.vams"
        module law_probe(in_a, in_b, code_a, code_b);
        input in_a, in_b;
        output code_a, code_b;
        electrical in_a, in_b, code_a, code_b;
        parameter real vth_a = 3.141592653e-02;
        parameter real vth_b = 1.505535214250;
        real seen_a, seen_b;
        analog begin
            @(initial_step) begin
                seen_a = -1.0;
                seen_b = -1.0;
            end
            @(cross(V(in_a) - vth_a, +1)) seen_a = $abstime;
            @(cross(V(in_b) - vth_b, +1)) seen_b = $abstime;
            V(code_a) <+ (seen_a >= 0.0) ? (seen_a - 3.141592653e-9) / 1p : -1.0;
            V(code_b) <+ (seen_b >= 0.0) ? (seen_b - 6.022140857e-9) / 1p : -1.0;
        end
        endmodule
    """)

    SCS_TEMPLATE = textwrap.dedent("""\
        simulator lang=spectre
        global 0
        Va (in_a 0) vsource type=pwl wave=[0 0 10n 0.1]
        Vb (in_b 0) vsource type=pwl wave=[0 0 10n 2.5]
        I0 (in_a in_b code_a code_b) law_probe
        simulatorOptions options reltol=1e-4 vabstol=1e-6{slack}
        tran tran stop=10n maxstep=20p errpreset=conservative
        save code_a code_b
        ahdl_include "law_probe.va"
    """)

    def _run(self, tmp_path, slack_opt, out_name):
        va_file = tmp_path / "law_probe.va"
        va_file.write_text(self.VA_SRC)
        scs_file = tmp_path / f"tb_{out_name}.scs"
        scs_file.write_text(self.SCS_TEMPLATE.format(slack=slack_opt))
        from evas.netlist.runner import evas_simulate
        ok = evas_simulate(str(scs_file), output_dir=str(tmp_path / out_name))
        assert ok
        rows = list(csv.DictReader((tmp_path / out_name / "tran.csv").open()))
        last = rows[-1]
        return float(last["code_a"]), float(last["code_b"])

    def test_law_mode_reproduces_spectre_lateness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVAS_ENGINE", "evas-rust")
        # Default: exact analytic event times.
        code_a, code_b = self._run(tmp_path, "", "out_default")
        assert abs(code_a) < 1e-3, f"default not exact: {code_a} ps"
        assert abs(code_b) < 1e-3, f"default not exact: {code_b} ps"
        # Law mode: Delta = 0.5 * reltol * t_root for a from-origin ramp.
        code_a, code_b = self._run(
            tmp_path, " evas_cross_acceptance_slack_factor=1.0", "out_law")
        assert code_a == pytest.approx(0.5 * 1e-4 * 3.141592653e-9 * 1e12, rel=0.02), code_a
        assert code_b == pytest.approx(0.5 * 1e-4 * 6.022140857e-9 * 1e12, rel=0.02), code_b


class TestCrossPhaseClassification:
    """V(n1, n2) <+ x drives n1 only; n2 is the reference. A cross expression
    that references a rail used solely as a contribution node2 (the common
    V(in, VSS) benchmark style) must stay pre-phase, or pre-phase-only
    features (cross-acceptance law mode) silently never apply."""

    def test_contribution_node2_is_not_contributed(self):
        from evas.compiler.parser import parse
        from evas.simulator.rust_program import (
            _collect_contributed_nodes,
            lower_stmt,
        )

        src = textwrap.dedent("""\
            `include "disciplines.vams"
            module phase_probe(inp, VSS, outp);
            input inp;
            inout VSS;
            output outp;
            electrical inp, VSS, outp;
            real q;
            analog begin
                @(initial_step) q = 0.0;
                @(cross(V(inp, VSS) - 0.5, +1)) q = 1.0;
                V(outp, VSS) <+ q;
            end
            endmodule
        """)
        module = parse(src)
        body_ir = lower_stmt(module.analog_block.body)
        contributed = _collect_contributed_nodes(body_ir)
        assert "outp" in contributed
        assert "VSS" not in contributed, (
            "contribution node2 wrongly counted as contributed: "
            f"{sorted(contributed)}"
        )
        assert "inp" not in contributed


class TestCadenceLrmGapFillRunnerAllowlist:
    """The netlist runner preflight must not lag backend language support."""

    def test_runner_accepts_cadence_lrm_gap_fill_helpers(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVAS_ENGINE", "python")
        va_file = tmp_path / "cadence_lrm_gap_runner.va"
        va_file.write_text(textwrap.dedent("""\
            `include "disciplines.vams"
            module cadence_lrm_gap_runner(inp, out);
            input voltage inp;
            output voltage out;
            parameter real gain = 2.0;
            parameter string table_file = "missing.tbl";
            real surf[0:1][0:1];
            real value;
            analog initial begin
                $analog_node_alias(out, inp);
            end
            analog begin
                V(out) : ddt(V(out)) == V(inp);
                value = $vt($temperature()) + $simparam("tnom", 27.0)
                      + $param_given(gain) + $port_connected(out)
                      + $rtoi(2.7) + $cds_get_mc_trial_number()
                      + inp.potential.abstol
                      + potential(inp)
                      + $table_model(0.5, table_file)
                      + $table_model(0.5, 0.25, surf)
                      + idt(I(out), 0.0)
                      + laplace_nd(V(inp), {1}, {1, 1})
                      + zi_nd(V(inp), {1}, {1, 1}, 1n);
                potential(out) <+ value;
            end
            endmodule
        """))
        scs_file = tmp_path / "tb_cadence_lrm_gap_runner.scs"
        scs_file.write_text(textwrap.dedent("""\
            simulator lang=spectre
            global 0
            ahdl_include "cadence_lrm_gap_runner.va"
            Vin (inp 0) vsource dc=0.2
            X0 (inp out) cadence_lrm_gap_runner
            tran tran stop=1n
            save inp out
        """))

        assert evas_simulate(str(scs_file), output_dir=str(tmp_path / "out"))
