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
    ):
        assert needle in serve.SYSTEM, f"SYSTEM missing: {needle}"
    print("SYSTEM pitfalls + run-44842138147 rules present: OK")


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
    print("OFFLINE SMOKE TEST PASSED")
