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
import textwrap
from pathlib import Path

import pytest

from evas.netlist.spectre_parser import (
    _extract_nodes,
    _normalize_node_name,
    parse_spectre,
)
from evas.netlist.runner import _add_spectre_source, SpectreSource
from evas.simulator.engine import Simulator


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

        scs_content = textwrap.dedent(f"""\
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


# ===========================================================================
# _add_spectre_source — degenerate cases
# ===========================================================================

def _make_pulse(name, val0, val1, period=0.0, **extra):
    params = {"type": "pulse", "val0": val0, "val1": val1, "period": period}
    params.update(extra)
    return SpectreSource(name=name, node_pos="clk", node_neg="0",
                         source_type="pulse", params=params)


def _make_sine(name, freq=0.0, ampl=0.0, sinedc=0.0):
    params = {"type": "sine", "freq": freq, "ampl": ampl, "sinedc": sinedc}
    return SpectreSource(name=name, node_pos="vin", node_neg="0",
                         source_type="sine", params=params)


class TestAddSpectreSourceDegenerateCases:

    def _sim(self):
        return Simulator()

    def test_pulse_no_period_warns_and_becomes_dc(self):
        """val0=0, val1=0, period not set → DC 0V + warning."""
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
        """val0≠val1 but period=0 → DC at val1 + warning."""
        src = _make_pulse("Vclk", val0=0.0, val1=1.8, period=0.0)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert len(warns) == 1
        assert "period not set" in warns[0]

    def test_pulse_valid_no_warnings(self):
        """Well-formed pulse → no warnings."""
        src = _make_pulse("Vclk", val0=0.0, val1=1.8, period=10e-9,
                          rise=50e-12, fall=50e-12)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert warns == []

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
        assert "ampl=0" in warns[0]

    def test_sine_valid_no_warnings(self):
        """Well-formed sine → no warnings."""
        src = _make_sine("Vin", freq=100e6, ampl=0.4, sinedc=0.45)
        sim = self._sim()
        warns = _add_spectre_source(sim, src, "0")
        assert warns == []
