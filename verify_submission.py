#!/usr/bin/env python3
"""Deterministic self-check for a PLE submission.zip (run 44842138147 fix).

Why this exists: run 44842138147 was graded with a zip the agent had only
checked structurally (sheets present). Its content was wrong in ways a
script catches in under a second:

- deck_code "B-36" is not a catalogue code (real codes: D22-36/4-2, ...),
  so the grader priced the deck at Infinity and the objective was 0;
- chord codes "W8x24" / "HSS6x6x1/4" are not in the chord catalogue
  (real codes: PL3x3/8, ..., PL12x1);
- the constraint register had 10 rows (7 judged), capping the paper score
  at ~0.03 no matter how good the design was;
- reported drift 0.12 in vs the grader's re-analysis 146,038 in.

Run this after EVERY pack step and fix every ERROR before finishing:
    python3 verify_submission.py --zip output/submission.zip \
        --deck-catalogue work/deck_catalogue.csv \
        --chord-catalogue work/chord_catalogue.csv \
        --documents work/corpus.txt [--deck-span 4]

The catalogue CSVs are dumps of the catalogue tables from the task
documents (deck: deck_code, nominal_shear_strength_plf,
shear_stiffness_G_prime_kip_per_in, installed_cost_usd_per_sf; chord:
section_code, ...). The corpus file lists one governing document name per
line (superseded revisions and other buildings' studies must NOT be in it).
ASD capacity is nominal / 2.35 (omega_d from the deck manual).

Exit 0 = PASS (warnings allowed), 1 = FAIL. Checks are structural and
arithmetic only: this script never grades engineering judgement.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import openpyxl

EXPECTED_FILES = (
    "lateral_schedule.xlsx",
    "diaphragm_plan.dxf",
    "constraint_register.xlsx",
    "calculations.pdf",
    "design_basis.md",
)

SCHEDULE_SHEETS = {
    "DeckZones": ("zone_id", "deck_code", "G_prime_used_kip_per_in",
                  "capacity_asd_plf", "area_sf", "cost_usd"),
    "ChordSchedule": ("chord_id", "section_code", "length_ft", "cost_usd"),
    "Piers": ("pier_id", "panel", "line", "s0_ft", "s1_ft", "width_ft",
              "thickness_in", "top_z_ft", "k_kip_per_in", "embeds", "tiedown"),
    "ShearWalls": ("scenario", "line", "line_load_plf", "reaction_kips"),
    "Diaphragm": ("scenario", "line", "station_frac", "unit_shear_plf"),
    "Chords": ("scenario", "chord_id", "force_kips"),
    "Drift": ("scenario", "direction", "deflection_in"),
    "Convergence": ("level", "mesh_size_ft", "elements",
                    "max_unit_shear_plf", "key_reaction_kips"),
    "Cost": ("item", "quantity", "unit_cost_usd", "total_usd"),
}

OMEGA_D = 2.35  # ASD safety factor for deck shear (deck manual note 2)
DRIFT_IMPLAUSIBLE_IN = 10.0  # sanity bound: a real drift is inches, not feet


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, check, detail):
        self.errors.append((check, detail))

    def warn(self, check, detail):
        self.warnings.append((check, detail))

    def ok(self):
        return not self.errors

    def print(self):
        for check, detail in self.errors:
            print(f"FAIL {check}: {detail}")
        for check, detail in self.warnings:
            print(f"WARN {check}: {detail}")
        print(f"{'PASS' if self.ok() else 'FAIL'}: "
              f"{len(self.errors)} errors, {len(self.warnings)} warnings")


def _num(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def _close(a, b, rel=0.01):
    if a is None or b is None:
        return False
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)


def _rows(ws):
    header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if all(v is None or v == "" for v in row):
            continue
        out.append(dict(zip(header, row)))
    return out


def _load_catalogue(path, key_col):
    table = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = (row.get(key_col) or "").strip()
            if key:
                table[key] = row
    return table


def _ramp_warning(report, sheet, rows, key_cols, val_col):
    """Warn when a value is an exact arithmetic ramp across scenarios.

    Real scenario results follow wind data (directions, coefficients);
    a perfect linear ramp in every column is a placeholder, not a
    calculation. This is a warning: a genuine ramp is possible, so the
    agent must judge it, not the script.
    """
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(c) for c in key_cols)
        groups[key].append((str(r.get("scenario")), _num(r.get(val_col))))
    for key, vals in groups.items():
        nums = [v for _, v in vals if v is not None]
        if len(nums) < 6:
            continue
        diffs = [round(b - a, 9) for a, b in zip(nums, nums[1:])]
        if diffs and len(set(diffs)) == 1 and diffs[0] != 0:
            report.warn(
                f"{sheet}.ramp",
                f"{key}: {val_col} is an exact linear ramp "
                f"({nums[0]}..{nums[-1]} step {diffs[0]}) across scenarios — "
                "recompute from scenario wind data, do not type a sequence",
            )
            return


def check_zip(path, report):
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        report.error("zip", f"cannot open {path}: {exc}")
        return None
    names = zf.namelist()
    for want in EXPECTED_FILES:
        if want not in names:
            report.error("zip", f"missing file: {want}")
    for name in names:
        if name not in EXPECTED_FILES:
            report.warn("zip", f"unexpected extra file: {name}")
    if len(names) != len(EXPECTED_FILES):
        report.warn("zip", f"expected exactly {len(EXPECTED_FILES)} files, got {len(names)}")
    return zf


def check_schedule(zf, deck, chord, deck_span, report):
    try:
        wb = openpyxl.load_workbook(zf.open("lateral_schedule.xlsx"), data_only=True)
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        report.error("schedule", f"cannot read lateral_schedule.xlsx: {exc}")
        return
    max_asd = 0.0
    for sheet, header in SCHEDULE_SHEETS.items():
        if sheet not in wb.sheetnames:
            report.error("schedule", f"missing sheet: {sheet}")
            continue
        got = tuple(c.value for c in next(wb[sheet].iter_rows(min_row=1, max_row=1)))
        if got != header:
            report.error("schedule", f"{sheet}: header {got} != {header}")
    if "DeckZones" in wb.sheetnames:
        rows = _rows(wb["DeckZones"])
        if not rows:
            report.error("deck", "DeckZones has no rows")
        for r in rows:
            code = str(r.get("deck_code") or "").strip()
            g, cap, area, cost = (_num(r.get(c)) for c in
                                  ("G_prime_used_kip_per_in", "capacity_asd_plf",
                                   "area_sf", "cost_usd"))
            if deck is None:
                report.warn("deck", "no deck catalogue given; code/value checks skipped")
                break
            row = deck.get(code)
            if row is None:
                report.error("deck.code",
                             f"{code!r} is not in the deck catalogue — copy the "
                             "code verbatim from the catalogue table; a series "
                             "or manual name is not a code")
                continue
            g_tab = _num(row.get("shear_stiffness_G_prime_kip_per_in"))
            if deck_span == 4 and g_tab is not None:
                g_tab *= 1.15  # manual note 4: 4 ft span stiffens G' by 1.15
            nominal = _num(row.get("nominal_shear_strength_plf"))
            rate = _num(row.get("installed_cost_usd_per_sf"))
            if g is None or not _close(g, g_tab, 0.01):
                report.error("deck.G_prime",
                             f"{code}: G' used {g} != catalogue {g_tab}")
            if cap is None or nominal is None or not _close(cap, nominal / OMEGA_D, 0.01):
                report.error("deck.capacity",
                             f"{code}: ASD capacity {cap} != nominal {nominal} "
                             f"/ {OMEGA_D} = {None if nominal is None else round(nominal / OMEGA_D, 1)}")
            if area is None or area <= 0:
                report.error("deck.area", f"{code}: area {area} must be > 0")
            if cost is None or rate is None or area is None or not _close(cost, area * rate, 0.01):
                report.error("deck.cost",
                             f"{code}: cost {cost} != area {area} x ${rate}/sf")
            if cap:
                max_asd = max(max_asd, cap)
    if "ChordSchedule" in wb.sheetnames:
        for r in _rows(wb["ChordSchedule"]):
            code = str(r.get("section_code") or "").strip()
            if chord is not None and code not in chord:
                report.error("chord.code",
                             f"{code!r} is not in the chord catalogue — copy "
                             "the section code verbatim from the catalogue table")
            if _num(r.get("length_ft")) is None or _num(r.get("length_ft")) <= 0:
                report.error("chord.length", f"{code}: length must be > 0")
            if _num(r.get("cost_usd")) is None or _num(r.get("cost_usd")) < 0:
                report.error("chord.cost", f"{code}: cost must be finite and >= 0")
    if "Piers" in wb.sheetnames:
        for r in _rows(wb["Piers"]):
            pid = r.get("pier_id")
            s0, s1, w = _num(r.get("s0_ft")), _num(r.get("s1_ft")), _num(r.get("width_ft"))
            if s0 is None or s1 is None or s0 >= s1:
                report.error("piers.extent", f"{pid}: need s0 < s1, got {s0}, {s1}")
            elif w is None or not _close(w, s1 - s0, 0.01):
                report.error("piers.width", f"{pid}: width {w} != s1-s0 = {s1 - s0}")
            for col in ("thickness_in", "top_z_ft", "k_kip_per_in"):
                v = _num(r.get(col))
                if v is None or v <= 0:
                    report.error("piers.value", f"{pid}: {col} = {r.get(col)} must be > 0")
            emb = r.get("embeds")
            if not isinstance(emb, int) and _num(emb) is None:
                report.error("piers.embeds", f"{pid}: embeds = {emb!r} must be an integer >= 0")
            if str(r.get("tiedown") or "").strip().lower() not in ("yes", "no"):
                report.error("piers.tiedown", f"{pid}: tiedown must be 'yes' or 'no'")
    for sheet, val_col in (("ShearWalls", "line_load_plf"), ("Diaphragm", "unit_shear_plf"),
                           ("Chords", "force_kips"), ("Drift", "deflection_in")):
        if sheet not in wb.sheetnames:
            continue
        rows = _rows(wb[sheet])
        for r in rows:
            v = _num(r.get(val_col))
            if v is None or v < 0:
                report.error(f"{sheet}.value", f"{r}: {val_col} = {r.get(val_col)} must be finite and >= 0")
        if sheet == "Diaphragm":
            for r in rows:
                v = _num(r.get(val_col))
                if max_asd and v is not None and v > max_asd:
                    report.error("diaphragm.capacity",
                                 f"{r}: unit shear {v} plf exceeds the largest "
                                 f"deck ASD capacity {max_asd} plf — infeasible")
        key_cols = ("line",) if sheet in ("ShearWalls", "Diaphragm") else (
            ("chord_id",) if sheet == "Chords" else ("direction",))
        _ramp_warning(report, sheet, rows, key_cols, val_col)
    if "Drift" in wb.sheetnames:
        for r in _rows(wb["Drift"]):
            v = _num(r.get("deflection_in"))
            if v is not None and v >= DRIFT_IMPLAUSIBLE_IN:
                report.error("drift.implausible",
                             f"{r}: deflection {v} in is implausible — a formula "
                             "or units error, not a real answer; recompute")
    if "Cost" in wb.sheetnames:
        rows = _rows(wb["Cost"])
        total = 0.0
        for r in rows:
            q, u, t = (_num(r.get(c)) for c in ("quantity", "unit_cost_usd", "total_usd"))
            if q is None or u is None or t is None or not _close(t, q * u, 0.01):
                report.error("cost.arithmetic",
                             f"{r.get('item')}: total {t} != quantity {q} x ${u}")
            elif t < 0:
                report.error("cost.negative", f"{r.get('item')}: negative total")
            else:
                total += t
        if total <= 0:
            report.error("cost.total", "Cost sheet sums to <= 0")
        if "DeckZones" in wb.sheetnames:
            deck_total = sum((_num(r.get("cost_usd")) or 0.0) for r in _rows(wb["DeckZones"]))
            deck_items = [r for r in rows if "deck" in str(r.get("item") or "").lower()]
            if deck_items:
                item_total = sum((_num(r.get("total_usd")) or 0.0) for r in deck_items)
                if not _close(item_total, deck_total, 0.02):
                    report.error("cost.deck",
                                 f"Cost deck total {item_total} != sum of "
                                 f"DeckZones costs {deck_total}")


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def check_register(zf, min_rows, corpus, report):
    try:
        wb = openpyxl.load_workbook(zf.open("constraint_register.xlsx"), data_only=True)
    except Exception as exc:  # noqa: BLE001
        report.error("register", f"cannot read constraint_register.xlsx: {exc}")
        return
    if "Register" not in wb.sheetnames:
        report.error("register", "missing sheet: Register")
        return
    ws = wb["Register"]
    header = tuple(c.value for c in next(ws.iter_rows(min_row=1, max_row=1)))
    want = ("document", "clause", "requirement", "value", "units", "items")
    if header != want:
        report.error("register", f"header {header} != {want}")
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True)
            if any(v is not None and v != "" for v in r)]
    if len(rows) < min_rows:
        report.error("register.rows",
                     f"only {len(rows)} rows (minimum {min_rows}): the paper "
                     "score counts a family only when it is satisfied AND "
                     "registered — a small register caps the score no matter "
                     "how good the design is")
    seen = set()
    for i, r in enumerate(rows, start=2):
        doc, clause = (str(r[0] or "").strip(), str(r[1] or "").strip())
        if not doc or not clause:
            report.error("register.empty", f"row {i}: document/clause must not be empty")
            continue
        if corpus is not None and doc not in corpus:
            report.error("register.document",
                         f"row {i}: {doc!r} is not in the governing corpus — "
                         "a superseded revision or another building's study "
                         "is a wrong row")
        parts = [p.strip() for p in clause.split(";") if p.strip()]
        if len(parts) > 3:
            report.error("register.clauses",
                         f"row {i}: {len(parts)} clauses, at most 3 per row")
        numbers = sum(len(_NUM_RE.findall(str(c))) for c in r)
        if numbers > 8:
            report.error("register.numbers",
                         f"row {i}: {numbers} numbers, at most 8 per row")
        key = (doc, clause)
        if key in seen:
            report.error("register.duplicate", f"row {i}: duplicate {key}")
        seen.add(key)


def check_dxf(zf, report):
    try:
        import ezdxf
    except ImportError:
        report.warn("dxf", "ezdxf not installed; DXF checks skipped")
        return
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as fh:
            fh.write(zf.read("diaphragm_plan.dxf"))
            tmp = fh.name
        try:
            doc = ezdxf.readfile(tmp)
        finally:
            import os
            os.unlink(tmp)
    except Exception as exc:  # noqa: BLE001
        report.error("dxf", f"cannot parse diaphragm_plan.dxf: {exc}")
        return
    msp = doc.modelspace()
    entities = list(msp)
    if not entities:
        report.error("dxf", "drawing has no entities")
        return
    polys = [e for e in entities if e.dxftype() in ("LWPOLYLINE", "POLYLINE")]
    if not polys:
        report.error("dxf", "no polyline footprint in the drawing")
    try:
        from ezdxf.bbox import extents as _extents
        ext = _extents(entities)
        size = ext.size
        if size.x <= 0 or size.y <= 0 or max(size.x, size.y) > 100000:
            report.error("dxf.extents",
                         f"drawing extents {size.x:.1f} x {size.y:.1f} ft look "
                         "wrong — the plan is drawn on the building grid in feet")
    except Exception as exc:  # noqa: BLE001
        report.warn("dxf", f"could not measure extents: {exc}")


def check_pdf(zf, report):
    try:
        data = zf.read("calculations.pdf")
    except KeyError:
        return  # already reported under zip
    text = None
    try:
        import fitz  # pymupdf
        with fitz.open(stream=data, filetype="pdf") as doc:
            pages = doc.page_count
            text = "\n".join(p.get_text() for p in doc)
    except ImportError:
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = len(reader.pages)
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except ImportError:
            report.warn("pdf", "no PDF reader installed; page/text checks skipped")
            return
    except Exception as exc:  # noqa: BLE001
        report.error("pdf", f"cannot parse calculations.pdf: {exc}")
        return
    if pages < 1:
        report.error("pdf", "calculations.pdf has no pages")
    if text is not None and len(text.strip()) < 200:
        report.error("pdf", "calculations.pdf has almost no text — check it rendered")


def check_basis(zf, report):
    try:
        text = zf.read("design_basis.md").decode("utf-8", "replace")
    except KeyError:
        return  # already reported under zip
    if len(text.strip()) < 500:
        report.warn("basis", "design_basis.md is very short — the narrative "
                             "must carry the required tables and every resolved "
                             "document conflict")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-check a PLE submission.zip.")
    ap.add_argument("--zip", required=True, dest="zip_path")
    ap.add_argument("--deck-catalogue")
    ap.add_argument("--chord-catalogue")
    ap.add_argument("--documents", help="one governing document name per line")
    ap.add_argument("--register-min-rows", type=int, default=60)
    ap.add_argument("--deck-span", type=int, choices=(4, 6), default=6,
                    help="deck span in ft for the G' adjustment (manual note 4)")
    args = ap.parse_args(argv)

    report = Report()
    deck = chord = corpus = None
    if args.deck_catalogue:
        try:
            deck = _load_catalogue(args.deck_catalogue, "deck_code")
        except OSError as exc:
            report.error("catalogue", f"cannot read deck catalogue: {exc}")
    if args.chord_catalogue:
        try:
            chord = _load_catalogue(args.chord_catalogue, "section_code")
        except OSError as exc:
            report.error("catalogue", f"cannot read chord catalogue: {exc}")
    if args.documents:
        try:
            corpus = {ln.strip() for ln in Path(args.documents).read_text().splitlines()
                      if ln.strip()}
        except OSError as exc:
            report.error("catalogue", f"cannot read documents list: {exc}")

    zf = check_zip(args.zip_path, report)
    if zf is not None:
        check_schedule(zf, deck, chord, args.deck_span, report)
        check_register(zf, args.register_min_rows, corpus, report)
        check_dxf(zf, report)
        check_pdf(zf, report)
        check_basis(zf, report)
    report.print()
    return 0 if report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
