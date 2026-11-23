"""Offline smoke test for serve.py (no model quota, no network).

Run:  python3 test_serve_offline.py   (use the venv with openpyxl/ezdxf
installed for the helper tests: ~/workspace/.tmp/ple-venv/bin/python)
Checks:
  1. The run-44702206886 pitfalls lines are present in SYSTEM.
  2. The run-44842138147 rules are present in SYSTEM (verbatim catalogue
     codes, big register, geometry from the IFC, re-package + verify).
  3. solve() still runs end to end with a fake model client that answers
     with a plain final message (exercises the loop + VERIFY_NUDGE path),
     and installs both harness helpers into the container.
  4. ifc_extract.py parses a synthetic IFC with the exact failure modes
     of run 44842138147 (comma inside a quoted description, property
     linking, placement chain).
  6. _repackage_probe() reports a zip stale exactly when a source file
     is newer than it, using a fake session whose exec runs locally.
  7. zones_extract.py reads a synthetic mini S-201 taskdir (2 zones, 2
     chord lines, 2 catalogue codes) and returns the exact scheme, and
     fails cleanly on an empty taskdir; on the real wind-diaphragm-shear
     fixture (when present) its four tables match the grader's truth.json.
  8. scheme_check.py PASSes a zip built from the mini scheme and FAILs a
     zip with invented IDs/codes (and the real 22 Nov zip, when present).
  9. solve() gates on the scheme: a fake model that tries to finish with
     a bad zip gets a SCHEME_NUDGE (finish-time gate) and a fake model
     that keeps working gets one at the round-5 checkpoint; after the zip
     is fixed the run proceeds to the verify nudge and finishes.
"""
import asyncio
import io
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("MODEL_NAME", "openai-compatible/nvidia/nemotron-3.5-super-vl:free")
# Dummy key only so openai.AsyncOpenAI() constructs at import; the client
# is replaced by a fake below and no network call is ever made.
os.environ.setdefault("OPENAI_API_KEY", "dummy-offline-test-key")

import serve
import taskbridge


def check_system():
    for needle in (
        "layer.dxf.name (not .name)",
        "entity.dxf.text (not .text)",
        "p['mark']",
        # Run 44842138147 rules.
        "copied character-for-character from the catalogue table",
        "at least 60 rows",
        "harness/ifc_extract.py",
        "harness/verify_submission.py",
        "re-run your pack script",
        # Run 44965326609 (fix-3) rules.
        "harness/zones.csv",
        "harness/chords.csv",
        "harness/scheme_check.py",
        "harness/zones_extract.py",
        "NO total row",
        "144 rows",
    ):
        assert needle in serve.SYSTEM, f"SYSTEM missing: {needle}"
    print("SYSTEM pitfalls + run-44842138147 rules + fix-3 scheme rules: OK")


class FakeMessage:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None

    def model_dump(self, exclude_none=True):
        return {"role": "assistant", "content": self.content}


class FakeChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "stop"


class FakeReply:
    def __init__(self, message):
        self.choices = [FakeChoice(message)]
        self.usage = None


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        # First call: deliverables exist (fail-open probe) so the loop sends
        # VERIFY_NUDGE; second call: finish.
        return FakeReply(FakeMessage("done: deliverables verified"))


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


class FakeSession:
    workdir = "/tmp"
    instruction = "do a thing"  # names no deliverable paths
    final_message = ""

    def __init__(self):
        self.written = {}

    def time_left(self):
        return None

    async def exec(self, command):
        raise taskbridge.TaskError("no container in offline test")

    async def write_file(self, path, data, append=False):
        self.written[path] = data
        return len(data)

    async def log(self, text):
        pass

    async def usage(self, *args):
        pass


async def check_loop():
    serve.client = FakeClient()
    session = FakeSession()
    result = await serve.solve(session)
    assert result == "done: deliverables verified", f"unexpected result: {result!r}"
    assert serve.client.chat.completions.calls == 2, (
        f"expected 2 model calls (verify nudge + finish), "
        f"got {serve.client.chat.completions.calls}"
    )
    for name in serve.HARNESS_HELPERS:
        assert f"/tmp/harness/{name}" in session.written, (
            f"helper {name} was not installed into the container"
        )
    print("solve() loop with fake model (verify nudge + finish + helpers): OK")


SYNTH_IFC = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#5=IFCPERSON($,$,'',$,$,$,$,$);
#10=IFCLOCALPLACEMENT($,#11);
#11=IFCAXIS2PLACEMENT3D(#12,$,$);
#12=IFCCARTESIANPOINT((0.,0.,0.));
#20=IFCLOCALPLACEMENT(#10,#21);
#21=IFCAXIS2PLACEMENT3D(#22,$,$);
#22=IFCCARTESIANPOINT((120.,0.,360.));
#30=IFCPRODUCTDEFINITIONSHAPE($,$,(#31));
#31=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#32));
#32=IFCEXTRUDEDAREASOLID(#33,#34,.Z.,120.);
#33=IFCRECTANGLEPROFILEDEF(.AREA.,'Panel profile',$,45.,8.);
#34=IFCAXIS2PLACEMENT3D(#35,$,$);
#35=IFCCARTESIANPOINT((0.,0.,0.));
#40=IFCWALL('guid-1',#5,'S01','Panel, south wall, type A',$ ,#20,#30,'TAG-S01');
#50=IFCPROPERTYSET('ps1',#5,'Pset_Panel',$,(#51,#52));
#51=IFCPROPERTYSINGLEVALUE('PanelMark',$,IFCLABEL('S01'),$);
#52=IFCPROPERTYSINGLEVALUE('BaseElevationMm',$,IFCREAL(0.0),$);
#60=IFCRELDEFINESBYPROPERTIES('rel1',#5,$,$,(#40),#50);
ENDSEC;
END-ISO-10303-21;
"""


def check_ifc_extract():
    import ifc_extract

    entities = ifc_extract.Parser(
        ifc_extract.tokenize(SYNTH_IFC)).parse_entities()
    walls = ifc_extract.extract_walls(entities)
    assert len(walls) == 1, f"expected 1 wall, got {len(walls)}"
    w = walls[0]
    assert w["mark"] == "S01", f"mark {w['mark']!r}"
    assert w["properties"].get("PanelMark") == "S01", "property not linked"
    assert w["properties"].get("BaseElevationMm") == 0.0, "typed value not unwrapped"
    assert w["placement_xyz"] == [120.0, 0.0, 360.0], f"placement {w['placement_xyz']}"
    assert w["profile"] == {"type": "rectangle", "x_dim": 45.0, "y_dim": 8.0}, (
        f"profile {w['profile']}")
    print("ifc_extract.py on synthetic IFC (comma in description, props, "
          "placement, profile): OK")


def _make_zip(path, deck_code, chord_code, register_rows):
    import openpyxl
    from openpyxl import Workbook
    import ezdxf

    wb = Workbook()
    ws = wb.active
    ws.title = "DeckZones"
    ws.append(["zone_id", "deck_code", "G_prime_used_kip_per_in",
               "capacity_asd_plf", "area_sf", "cost_usd"])
    ws.append(["Z1", deck_code, 17.8, round(980 / 2.35, 2), 43200, 43200 * 3.05])
    ws = wb.create_sheet("ChordSchedule")
    ws.append(["chord_id", "section_code", "length_ft", "cost_usd"])
    ws.append(["C1", chord_code, 240, 10800])
    ws = wb.create_sheet("Piers")
    ws.append(["pier_id", "panel", "line", "s0_ft", "s1_ft", "width_ft",
               "thickness_in", "top_z_ft", "k_kip_per_in", "embeds", "tiedown"])
    ws.append(["P001", "Panel-W", "W", 0, 45, 45, 8, 30, 1200, 4, "no"])
    for name, header, rows in (
        ("ShearWalls", ["scenario", "line", "line_load_plf", "reaction_kips"],
         [["W01", "W", 121.3, 21.6], ["W02", "W", 187.9, 33.1]]),
        ("Diaphragm", ["scenario", "line", "station_frac", "unit_shear_plf"],
         [["W01", "W", 0.25, 151.7], ["W02", "W", 0.5, 203.4]]),
        ("Chords", ["scenario", "chord_id", "force_kips"],
         [["W01", "C1", 8.3], ["W02", "C1", 14.7]]),
        ("Drift", ["scenario", "direction", "deflection_in"],
         [["W01", "X", 0.12], ["W02", "X", 0.31]]),
        ("Convergence", ["level", "mesh_size_ft", "elements",
                          "max_unit_shear_plf", "key_reaction_kips"],
         [["L1", 10, 432, 210, 48], ["L2", 5, 1728, 195, 50]]),
        ("Cost", ["item", "quantity", "unit_cost_usd", "total_usd"],
         [["Roof deck (Z1)", 43200, 3.05, 43200 * 3.05],
          ["Chords & collectors", 940, 45, 42300]]),
    ):
        ws = wb.create_sheet(name)
        ws.append(header)
        for r in rows:
            ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    wb2 = Workbook()
    ws = wb2.active
    ws.title = "Register"
    ws.append(["document", "clause", "requirement", "value", "units", "items"])
    for i in range(1, register_rows + 1):
        ws.append(["Criteria Rev 2", f"4.{i}", f"Requirement {i}", 1, "x", "All"])
    buf2 = io.BytesIO()
    wb2.save(buf2)
    doc = ezdxf.new("R2010")
    doc.modelspace().add_lwpolyline([(0, 0), (240, 0), (240, 180), (0, 180), (0, 0)])
    dxfbuf = io.StringIO()
    doc.write(dxfbuf)
    import fitz
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 720),
                        ("Calculation package.\n" * 20), fontsize=11)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("lateral_schedule.xlsx", buf.getvalue())
        z.writestr("diaphragm_plan.dxf", dxfbuf.getvalue().encode())
        z.writestr("constraint_register.xlsx", buf2.getvalue())
        z.writestr("calculations.pdf", pdf.tobytes())
        z.writestr("design_basis.md", "# Design basis\n\n" + "Prose. " * 200)
    return path


def check_verify_submission():
    try:
        import openpyxl  # noqa: F401
        import ezdxf  # noqa: F401
        import fitz  # noqa: F401
    except ImportError as exc:
        print(f"verify_submission tests SKIPPED (missing lib: {exc})")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        deck = os.path.join(tmp, "deck.csv")
        chord = os.path.join(tmp, "chord.csv")
        with open(deck, "w") as fh:
            fh.write("deck_code,gauge,nominal_shear_strength_plf,"
                     "shear_stiffness_G_prime_kip_per_in,installed_cost_usd_per_sf\n"
                     "D22-36/4-2,22,980,17.8,3.05\n")
        with open(chord, "w") as fh:
            fh.write("section_code,area_in2,weight_plf,allowable_tension_kips,"
                     "allowable_compression_kips\nPL6x1/2,3,10.2,90,54\n")
        bad = _make_zip(os.path.join(tmp, "bad.zip"), "B-36", "W8x24", 10)
        good = _make_zip(os.path.join(tmp, "good.zip"), "D22-36/4-2", "PL6x1/2", 60)
        for path, want in ((bad, 1), (good, 0)):
            proc = subprocess.run(
                [sys.executable, os.path.join(here, "verify_submission.py"),
                 "--zip", path, "--deck-catalogue", deck,
                 "--chord-catalogue", chord],
                capture_output=True, text=True, timeout=120)
            out = proc.stdout + proc.stderr
            if want == 1:
                assert proc.returncode == 1, f"{path} should FAIL:\n{out}"
                for needle in ("deck.code", "chord.code", "register.rows"):
                    assert needle in out, f"{path}: missing {needle}:\n{out}"
            else:
                assert proc.returncode == 0, f"{path} should PASS:\n{out}"
    print("verify_submission.py FAILs the run-44842138147 defects and PASSes "
          "the corrected zip: OK")


class LocalExecSession(FakeSession):
    """Fake session whose exec runs the command on this machine."""

    async def exec(self, command):
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        out, _ = await proc.communicate()
        return {"output": out.decode(), "exit_code": proc.returncode}


def _make_mini_taskdir(base):
    """Synthetic S-201 taskdir: 2 zones (Z22 spans 4 ft), 2 chord lines,
    2 deck codes, 2 chord codes — small enough to assert exactly."""
    import ezdxf

    taskdir = Path(base) / "task"
    (taskdir / "drawings").mkdir(parents=True)
    (taskdir / "procurement").mkdir()
    doc = ezdxf.new("R2010")
    for layer in ("S-DECK-ZONE", "S-CHORD", "S-COLL", "S-ANNO"):
        doc.layers.add(layer)
    msp = doc.modelspace()
    for x0 in (0, 60):
        msp.add_lwpolyline(
            [(x0, 0), (x0 + 60, 0), (x0 + 60, 60), (x0, 60)],
            close=True, dxfattribs={"layer": "S-DECK-ZONE"})
    # Outline: same shape as the two rects together (area 7200 -> a cut).
    msp.add_lwpolyline(
        [(0, 0), (120, 0), (120, 60), (0, 60)],
        close=True, dxfattribs={"layer": "S-DECK-ZONE"})
    msp.add_text("Z11", dxfattribs={"layer": "S-DECK-ZONE"}).set_placement((30, 30))
    msp.add_text("Z22", dxfattribs={"layer": "S-DECK-ZONE"}).set_placement((90, 30))
    msp.add_text("C-S", dxfattribs={"layer": "S-CHORD"}).set_placement((60, 5))
    msp.add_text("K-N2", dxfattribs={"layer": "S-COLL"}).set_placement((115, 30))
    msp.add_line((0, 0), (120, 0), dxfattribs={"layer": "S-CHORD"})
    msp.add_line((120, 0), (120, 60), dxfattribs={"layer": "S-COLL"})
    msp.add_text("ZONE Z22 NEAR THE GENERATOR YARD DECK SPAN 4 FT",
                 dxfattribs={"layer": "S-ANNO"}).set_placement((0, 100))
    doc.saveas(taskdir / "drawings" / "S-201_Roof_Framing_Plan.dxf")

    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Table 3"
    ws.append(["deck_code", "gauge", "nominal_shear_strength_plf",
               "shear_stiffness_G_prime_kip_per_in",
               "installed_cost_usd_per_sf"])
    ws.append(["D22-36/4-2", 22, 980, 17.8, 3.05])
    ws.append(["D16-36/4-2", 16, 700, 12.0, 2.50])
    notes = wb.create_sheet("Notes")
    notes.append(["no", "note"])
    notes.append([1, "omega_d = 2.35."])
    notes.append([2, "For a 4 ft deck span multiply the tabulated G' by 1.15."])
    wb.save(taskdir / "procurement" / "CDS-B36_Diaphragm_Design_Manual.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = "Table 1"
    ws.append(["section_code", "area_in2", "weight_plf",
               "allowable_tension_kips", "allowable_compression_kips"])
    ws.append(["PL3x3/8", 1.0, 2.5, 10, 8])
    ws.append(["PL6x1/2", 3.0, 10.2, 90, 54])
    notes = wb.create_sheet("Notes")
    notes.append(["no", "note"])
    notes.append([1, "Chord steel costs $2.85 per pound installed."])
    wb.save(taskdir / "procurement" / "KRS-CH_Chord_and_Collector_Schedule.xlsx")
    return taskdir


def _read_csv(path):
    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def check_zones_extract():
    import json

    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        taskdir = _make_mini_taskdir(tmp)
        out = os.path.join(tmp, "scheme")
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "zones_extract.py"),
             "--taskdir", str(taskdir), "--out", out],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, f"zones_extract failed:\n{proc.stdout}\n{proc.stderr}"
        zones = _read_csv(os.path.join(out, "zones.csv"))
        assert [(z["zone_id"], z["area_sf"], z["span_ft"]) for z in zones] == [
            ("Z11", "3600.0", "6"), ("Z22", "3600.0", "4")], zones
        chords = _read_csv(os.path.join(out, "chords.csv"))
        assert [(c["line_id"], c["axis"], c["length_ft"], c["role"])
                for c in chords] == [
            ("C-S", "X", "120.0", "chord"), ("K-N2", "Y", "60.0", "collector")], chords
        deck = _read_csv(os.path.join(out, "deck_catalogue.csv"))
        assert [(d["deck_code"], d["asd_plf"]) for d in deck] == [
            ("D22-36/4-2", "417.0"), ("D16-36/4-2", "297.9")], deck
        chord = _read_csv(os.path.join(out, "chord_catalogue.csv"))
        assert [c["section_code"] for c in chord] == ["PL3x3/8", "PL6x1/2"], chord
        meta = json.load(open(os.path.join(out, "meta.json")))
        assert (meta["omega_d"], meta["span_factor_4ft"],
                meta["chord_per_lb"]) == (2.35, 1.15, 2.85), meta
        # Empty taskdir: clean failure, exit 1, no crash.
        empty = os.path.join(tmp, "empty")
        os.makedirs(empty)
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "zones_extract.py"),
             "--taskdir", empty, "--out", os.path.join(tmp, "scheme2")],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 1, "empty taskdir should exit 1"
    print("zones_extract.py on synthetic mini taskdir (exact scheme) + "
          "empty-dir failure: OK")
    # Real fixture: the four tables must match the grader's truth exactly.
    run_dir = Path.home() / ("workspace/goals/ple-agent-harness-kaggle-400k/"
        "hidden_files/smoke-2026-11-22/artifacts/ple-run-wind-diaphragm-shear/"
        "ple-harness-20261122-001526")
    real_task = (run_dir / "tasks/wind-diaphragm-shear/environment/filesystem/"
        "final_batch4__structural_task5_wind_diaphragm_shear__blind")
    truth_path = (run_dir / "tasks/wind-diaphragm-shear/tests/corrino/"
        "structural_task5_wind_diaphragm_shear/grading/data/truth.json")
    if not (real_task.exists() and truth_path.exists()):
        print("zones_extract real-fixture check SKIPPED (artifact absent)")
        return
    truth = json.load(open(truth_path))
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "scheme")
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "zones_extract.py"),
             "--taskdir", str(real_task), "--out", out],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, f"real fixture failed:\n{proc.stdout}\n{proc.stderr}"
        zones = _read_csv(os.path.join(out, "zones.csv"))
        assert [z["zone_id"] for z in zones] == [z["id"] for z in truth["zones"]]
        assert [float(z["area_sf"]) for z in zones] == [
            float(z["area_sf"]) for z in truth["zones"]]
        chords = _read_csv(os.path.join(out, "chords.csv"))
        # Order differs (extract sorts alphabetically, truth is in drawing
        # order); compare as sets keyed by line id.
        assert {c["line_id"]: float(c["length_ft"]) for c in chords} == {
            c["id"]: float(c["s1"] - c["s0"]) for c in truth["chord_lines"]}
        deck = _read_csv(os.path.join(out, "deck_catalogue.csv"))
        assert {d["deck_code"] for d in deck} == {
            d["code"] for d in truth["deck_catalogue"]}
        chord = _read_csv(os.path.join(out, "chord_catalogue.csv"))
        assert {c["section_code"] for c in chord} == {
            c["code"] for c in truth["chord_catalogue"]}
    print("zones_extract.py on the real fixture matches truth.json "
          "(12 zones, 12 lines, 16 deck, 10 chord): OK")


def _make_scheme_zip(path, scheme_dir):
    """Full 5-file zip whose schedule follows the scheme CSVs exactly."""
    import openpyxl
    from openpyxl import Workbook
    import ezdxf

    zones = _read_csv(os.path.join(scheme_dir, "zones.csv"))
    chords = _read_csv(os.path.join(scheme_dir, "chords.csv"))
    deck = {d["deck_code"]: d
            for d in _read_csv(os.path.join(scheme_dir, "deck_catalogue.csv"))}
    chord = {c["section_code"]: c
             for c in _read_csv(os.path.join(scheme_dir, "chord_catalogue.csv"))}
    meta = __import__("json").load(open(os.path.join(scheme_dir, "meta.json")))
    code = "D22-36/4-2"
    d = deck[code]
    wb = Workbook()
    ws = wb.active
    ws.title = "DeckZones"
    ws.append(["zone_id", "deck_code", "G_prime_used_kip_per_in",
               "capacity_asd_plf", "area_sf", "cost_usd"])
    deck_total = 0.0
    for z in zones:
        g = float(d["shear_stiffness_G_prime_kip_per_in"])
        if int(z["span_ft"]) == 4:
            g *= meta["span_factor_4ft"]
        cost = float(z["area_sf"]) * float(d["installed_cost_usd_per_sf"])
        deck_total += cost
        ws.append([z["zone_id"], code, round(g, 2), float(d["asd_plf"]),
                   float(z["area_sf"]), round(cost, 2)])
    ws = wb.create_sheet("ChordSchedule")
    ws.append(["chord_id", "section_code", "length_ft", "cost_usd"])
    chord_total = 0.0
    for c in chords:
        cc = chord["PL3x3/8"]
        cost = (float(c["length_ft"]) * float(cc["weight_plf"])
                * meta["chord_per_lb"])
        chord_total += cost
        ws.append([c["line_id"], "PL3x3/8", float(c["length_ft"]), round(cost, 2)])
    for name, header, rows in (
        ("Piers", ["pier_id", "panel", "line", "s0_ft", "s1_ft", "width_ft",
                    "thickness_in", "top_z_ft", "k_kip_per_in", "embeds",
                    "tiedown"],
         [["P001", "Panel-W", "W", 0, 45, 45, 8, 30, 1200, 4, "no"]]),
        ("ShearWalls", ["scenario", "line", "line_load_plf", "reaction_kips"],
         [["W01", "W", 121.3, 21.6]]),
        ("Diaphragm", ["scenario", "line", "station_frac", "unit_shear_plf"],
         [["W01", "W", 0.25, 151.7]]),
        ("Chords", ["scenario", "chord_id", "force_kips"],
         [["W01", "C-S", 8.3]]),
        ("Drift", ["scenario", "direction", "deflection_in"],
         [["W01", "X", 0.12]]),
        ("Convergence", ["level", "mesh_size_ft", "elements",
                          "max_unit_shear_plf", "key_reaction_kips"],
         [["L1", 10, 432, 210, 48]]),
    ):
        ws = wb.create_sheet(name)
        ws.append(header)
        for r in rows:
            ws.append(r)
    ws = wb.create_sheet("Cost")
    ws.append(["item", "quantity", "unit_cost_usd", "total_usd"])
    for z in zones:
        ws.append([f"Roof deck {z['zone_id']}", float(z["area_sf"]),
                   float(d["installed_cost_usd_per_sf"]),
                   round(float(z["area_sf"])
                         * float(d["installed_cost_usd_per_sf"]), 2)])
    ws.append(["Chords & collectors", 1, round(chord_total, 2),
               round(chord_total, 2)])
    buf = io.BytesIO()
    wb.save(buf)
    wb2 = Workbook()
    ws = wb2.active
    ws.title = "Register"
    ws.append(["document", "clause", "requirement", "value", "units", "items"])
    buf2 = io.BytesIO()
    wb2.save(buf2)
    doc = ezdxf.new("R2010")
    doc.modelspace().add_lwpolyline([(0, 0), (240, 0), (240, 180), (0, 180), (0, 0)])
    dxfbuf = io.StringIO()
    doc.write(dxfbuf)
    import fitz
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_textbox(fitz.Rect(72, 72, 540, 720),
                        ("Calculation package.\n" * 20), fontsize=11)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("lateral_schedule.xlsx", buf.getvalue())
        z.writestr("diaphragm_plan.dxf", dxfbuf.getvalue().encode())
        z.writestr("constraint_register.xlsx", buf2.getvalue())
        z.writestr("calculations.pdf", pdf.tobytes())
        z.writestr("design_basis.md", "# Design basis\n\n" + "Prose. " * 200)
    return path


def _run_scheme_check(here, zip_path, scheme_dir):
    return subprocess.run(
        [sys.executable, os.path.join(here, "scheme_check.py"),
         "--zip", zip_path,
         "--zones", os.path.join(scheme_dir, "zones.csv"),
         "--chords", os.path.join(scheme_dir, "chords.csv"),
         "--deck-catalogue", os.path.join(scheme_dir, "deck_catalogue.csv"),
         "--chord-catalogue", os.path.join(scheme_dir, "chord_catalogue.csv"),
         "--meta", os.path.join(scheme_dir, "meta.json")],
        capture_output=True, text=True, timeout=120)


def check_scheme_check():
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        taskdir = _make_mini_taskdir(tmp)
        scheme_dir = os.path.join(tmp, "scheme")
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "zones_extract.py"),
             "--taskdir", str(taskdir), "--out", scheme_dir],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        good = _make_scheme_zip(os.path.join(tmp, "good.zip"), scheme_dir)
        proc = _run_scheme_check(here, good, scheme_dir)
        assert proc.returncode == 0, (
            f"correct-scheme zip should PASS:\n{proc.stdout}\n{proc.stderr}")
        assert "SCHEME PASS" in proc.stdout
        bad = _make_zip(os.path.join(tmp, "bad.zip"), "B-36", "W8x24", 10)
        proc = _run_scheme_check(here, bad, scheme_dir)
        assert proc.returncode == 1, (
            f"invented-scheme zip should FAIL:\n{proc.stdout}\n{proc.stderr}")
        assert "SCHEME FAIL" in proc.stdout and "ERR " in proc.stdout, proc.stdout
    print("scheme_check.py PASSes the correct-scheme zip and FAILs the "
          "invented-scheme zip: OK")
    run_dir = Path.home() / ("workspace/goals/ple-agent-harness-kaggle-400k/"
        "hidden_files/smoke-2026-11-22/artifacts/ple-run-wind-diaphragm-shear/"
        "ple-harness-20261122-001526")
    real_zip = (run_dir / "artifacts/wind-diaphragm-shear/files/filesystem/"
        "final_batch4__structural_task5_wind_diaphragm_shear__blind/output/"
        "submission.zip")
    if not real_zip.exists():
        print("scheme_check real-zip check SKIPPED (artifact absent)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        real_task = (run_dir / "tasks/wind-diaphragm-shear/environment/"
            "filesystem/final_batch4__structural_task5_wind_diaphragm_shear__blind")
        scheme_dir = os.path.join(tmp, "scheme")
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "zones_extract.py"),
             "--taskdir", str(real_task), "--out", scheme_dir],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        proc = _run_scheme_check(here, str(real_zip), scheme_dir)
        assert proc.returncode == 1 and "ERR " in proc.stdout, (
            f"real 22 Nov zip should FAIL the scheme check:\n{proc.stdout}")
    print("scheme_check.py FAILs the real 22 Nov zip (13 scheme errors): OK")


class DiskSession(LocalExecSession):
    """Fake session whose write_file lands on disk, so the harness probes
    (zones_extract at run start, scheme_check in the loop) run for real."""

    async def write_file(self, path, data, append=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "ab" if append else "wb"
        with open(path, mode) as fh:
            fh.write(data.encode() if isinstance(data, str) else data)
        self.written[path] = data
        return len(data)


class ScriptedMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=True):
        out = {"role": "assistant"}
        if self.content is not None:
            out["content"] = self.content
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        return out


class ScriptedCompletions:
    def __init__(self, replies):
        self.replies = replies
        self.calls = 0
        self.seen = []  # message lists passed to each create()

    async def create(self, **kwargs):
        self.calls += 1
        self.seen.append(kwargs.get("messages", []))
        return FakeReply(self.replies[self.calls - 1])


class ScriptedChat:
    def __init__(self, replies):
        self.completions = ScriptedCompletions(replies)


class ScriptedClient:
    def __init__(self, replies):
        self.chat = ScriptedChat(replies)


def _bash_tool(command, call_id="call_1"):
    import json as _json
    return ScriptedMessage(None, tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": "bash",
                     "arguments": _json.dumps({"command": command})},
    }])


def _scheme_nudges_seen(client):
    # The history accumulates, so the LAST messages list holds every nudge
    # sent during the run, each exactly once.
    if not client.chat.completions.seen:
        return []
    return [m["content"] for m in client.chat.completions.seen[-1]
            if isinstance(m.get("content"), str)
            and "FAILS the scheme check" in m["content"]]


async def check_scheme_gate_finish():
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        taskdir = _make_mini_taskdir(tmp)
        scheme_dir = os.path.join(tmp, "scheme")
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "zones_extract.py"),
             "--taskdir", str(taskdir), "--out", scheme_dir],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        outdir = taskdir / "output"
        outdir.mkdir()
        bad_zip = outdir / "submission.zip"
        _make_zip(str(bad_zip), "B-36", "W8x24", 10)
        good_zip = os.path.join(tmp, "good.zip")
        _make_scheme_zip(good_zip, scheme_dir)
        # The loop's probes run bare `python3` (as in the container); point
        # it at this venv so ezdxf/openpyxl are importable.
        venv_bin = os.path.dirname(sys.executable)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = venv_bin + os.pathsep + old_path
        # Fresh mtime, so the deliverable guard sees the zip as present.
        os.utime(bad_zip, None)
        session = DiskSession()
        session.workdir = tmp
        session.instruction = (
            f"You are working in `{taskdir}`. "
            "Write output/submission.zip there.")
        replies = [
            ScriptedMessage("done: first attempt"),
            _bash_tool(f"cp {good_zip} {bad_zip}"),
            ScriptedMessage("fixed: scheme corrected"),
            ScriptedMessage("verified: all good"),
        ]
        client = ScriptedClient(replies)
        serve.client = client
        try:
            result = await serve.solve(session)
        finally:
            os.environ["PATH"] = old_path
        assert result == "verified: all good", f"unexpected result: {result!r}"
        assert client.chat.completions.calls == 4, (
            f"expected 4 model calls (finish, fix, finish, finish), got "
            f"{client.chat.completions.calls}")
        nudges = _scheme_nudges_seen(client)
        assert len(nudges) == 1, f"expected 1 scheme nudge, got {len(nudges)}"
        assert "B-36" in nudges[0] or "Z1" in nudges[0], nudges[0]
    print("solve() finish-time scheme gate (FAIL -> nudge -> fix -> "
          "verify -> finish): OK")


async def check_scheme_gate_checkpoint():
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        taskdir = _make_mini_taskdir(tmp)
        scheme_dir = os.path.join(tmp, "scheme")
        proc = subprocess.run(
            [sys.executable, os.path.join(here, "zones_extract.py"),
             "--taskdir", str(taskdir), "--out", scheme_dir],
            capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        outdir = taskdir / "output"
        outdir.mkdir()
        bad_zip = outdir / "submission.zip"
        _make_zip(str(bad_zip), "B-36", "W8x24", 10)
        # The loop's probes run bare `python3` (as in the container); point
        # it at this venv so ezdxf/openpyxl are importable.
        venv_bin = os.path.dirname(sys.executable)
        old_path = os.environ.get("PATH", "")
        os.environ["PATH"] = venv_bin + os.pathsep + old_path
        # Fresh mtime, so the deliverable guard sees the zip as present.
        os.utime(bad_zip, None)
        session = DiskSession()
        session.workdir = tmp
        session.instruction = (
            f"You are working in `{taskdir}`. "
            "Write output/submission.zip there.")
        # 5 no-op tool rounds trigger the round-5 checkpoint (scheme FAIL,
        # nudge 1); the model never fixes it, so the finish gate spends
        # nudges 2 and 3, then the budget is out and the verify nudge fires.
        replies = ([_bash_tool("true", f"call_{i}") for i in range(1, 6)]
                   + [ScriptedMessage("will fix"),
                      ScriptedMessage("will fix"),
                      ScriptedMessage("will fix"),
                      ScriptedMessage("verified")])
        client = ScriptedClient(replies)
        serve.client = client
        try:
            result = await serve.solve(session)
        finally:
            os.environ["PATH"] = old_path
        assert result == "verified", f"unexpected result: {result!r}"
        assert client.chat.completions.calls == 9, (
            f"expected 9 model calls, got {client.chat.completions.calls}")
        nudges = _scheme_nudges_seen(client)
        assert len(nudges) == 3, f"expected 3 scheme nudges, got {len(nudges)}"
    print("solve() checkpoint scheme gate (round-5 nudge + finish-gate "
          "budget of 3, then verify): OK")


async def check_repackage_probe():
    with tempfile.TemporaryDirectory() as tmp:
        taskdir = Path(tmp) / "task"
        (taskdir / "output").mkdir(parents=True)
        (taskdir / "work").mkdir()
        zip_path = taskdir / "output" / "submission.zip"
        zip_path.write_bytes(b"zip")
        os.utime(zip_path, (1_000_000, 1_000_000))
        instruction = (f"You are working in `{taskdir}`. "
                       "Write output/submission.zip there.")
        session = LocalExecSession()
        session.workdir = tmp
        stale, mtime = await serve._repackage_probe(session, instruction)
        assert stale is False and mtime == 1_000_000, (stale, mtime)
        # A newer source file outside output/ makes the zip stale.
        src = taskdir / "work" / "build.py"
        src.write_text("x = 1\n")
        os.utime(src, (1_000_100, 1_000_100))
        stale, mtime = await serve._repackage_probe(session, instruction)
        assert stale is True and mtime == 1_000_000, (stale, mtime)
        # Rewriting the zip clears the staleness.
        zip_path.write_bytes(b"zip2")
        os.utime(zip_path, (1_000_200, 1_000_200))
        stale, mtime = await serve._repackage_probe(session, instruction)
        assert stale is False and mtime == 1_000_200, (stale, mtime)
    print("_repackage_probe() stale/clear behaviour: OK")


if __name__ == "__main__":
    check_system()
    asyncio.run(check_loop())
    check_ifc_extract()
    check_verify_submission()
    asyncio.run(check_repackage_probe())
    check_zones_extract()
    check_scheme_check()
    asyncio.run(check_scheme_gate_finish())
    asyncio.run(check_scheme_gate_checkpoint())
    print("OFFLINE SMOKE TEST PASSED")
