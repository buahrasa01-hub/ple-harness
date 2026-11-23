# scheme_check.py — deterministic scheme gate for the lateral package.
#
# Run 44965326609 (22 Nov): the agent's zip had 2 DeckZones rows with
# invented IDs (D1/D2) and invented codes (CDS-B36-3/2), and 6 ChordSchedule
# rows with invented IDs (C-N, COLL-*) and invented codes (CH-10/CH-8).
# The grader needs the 12 S-201 zones (Z11..Z43) with their true areas and
# catalogue codes, and the 12 chord/collector/opening lines with their true
# lengths and catalogue sections. This script checks exactly that, against
# the CSVs zones_extract.py writes from the task documents themselves.
#
# It checks structure and catalogue consistency only — never the
# engineering (loads, shears, deflections stay the agent's job). Exit 0
# PASS, 1 FAIL (errors printed, one per line), 2 inputs unavailable.
"""Check a lateral-package zip against the extracted S-201 scheme."""

import argparse
import csv
import json
import math
import os
import re
import sys


def _close(got, want, rel, absolute):
    if got is None or want is None:
        return False
    return abs(got - want) <= max(rel * abs(want), absolute)


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _sheet_from_zip(zip_path, name):
    import openpyxl
    import tempfile
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        target = None
        for n in zf.namelist():
            if n.lower().endswith("lateral_schedule.xlsx"):
                target = n
                break
        if target is None:
            return None
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tf:
            tf.write(zf.read(target))
            tmp = tf.name
    try:
        return _sheet(tmp, name)
    finally:
        os.unlink(tmp)


def _sheet(path, name):
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    for ws in wb.worksheets:
        if ws.title.lower().replace(" ", "") == name.lower().replace(" ", ""):
            rows = []
            header = None
            for row in ws.iter_rows(values_only=True):
                if header is None:
                    header = [str(c).strip().lower() if c is not None else ""
                              for c in row]
                    continue
                if all(v is None for v in row):
                    continue
                rows.append(dict(zip(header, row)))
            wb.close()
            return rows
    wb.close()
    return None


def check(zip_path, zones_csv, chords_csv, deck_csv, chord_csv, meta_path):
    errors = []
    zones = {r["zone_id"]: r for r in _read_csv(zones_csv)}
    chords = {r["line_id"]: r for r in _read_csv(chords_csv)}
    deck_cat = {r["deck_code"]: r for r in _read_csv(deck_csv)}
    chord_cat = {r["section_code"]: r for r in _read_csv(chord_csv)}
    meta = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    omega = float(meta.get("omega_d", 2.35))
    span_factor = float(meta.get("span_factor_4ft", 1.15))
    chord_per_lb = float(meta.get("chord_per_lb", 2.85))

    deck_rows = _sheet_from_zip(zip_path, "DeckZones")
    if deck_rows is None:
        return ["DeckZones sheet missing"]
    chord_rows = _sheet_from_zip(zip_path, "ChordSchedule")
    if chord_rows is None:
        return ["ChordSchedule sheet missing"]

    seen_zones, deck_cost_total = set(), 0.0
    for r in deck_rows:
        zid = str(r.get("zone_id") or "").strip()
        if not zid:
            continue
        seen_zones.add(zid)
        z = zones.get(zid)
        if z is None:
            errors.append(f"DeckZones: zone {zid!r} is not an S-201 zone "
                          f"(want one row per zone: {', '.join(sorted(zones))})")
            continue
        area = float(z["area_sf"])
        if not _close(_num(r.get("area_sf")), area, 0.015, 1.0):
            errors.append(f"DeckZones {zid}: area {r.get('area_sf')} vs {area}")
        code = str(r.get("deck_code") or "").strip()
        cat = deck_cat.get(code)
        if cat is None:
            errors.append(f"DeckZones {zid}: deck_code {code!r} not in the "
                          f"deck catalogue (valid: {', '.join(sorted(deck_cat))})")
            continue
        g_want = float(cat["shear_stiffness_G_prime_kip_per_in"]) * (
            span_factor if int(z["span_ft"]) == 4 else 1.0)
        if not _close(_num(r.get("g_prime_used_kip_per_in")), g_want, 0.02, 0.2):
            errors.append(f"DeckZones {zid}: G' {r.get('g_prime_used_kip_per_in')} "
                          f"vs catalogue {g_want:.1f} for {code}")
        asd_want = float(cat["nominal_shear_strength_plf"]) / omega
        if not _close(_num(r.get("capacity_asd_plf")), asd_want, 0.02, 5.0):
            errors.append(f"DeckZones {zid}: ASD {r.get('capacity_asd_plf')} "
                          f"vs catalogue {asd_want:.1f} for {code}")
        cost_want = area * float(cat["installed_cost_usd_per_sf"])
        if not _close(_num(r.get("cost_usd")), cost_want, 0.01, 1.0):
            errors.append(f"DeckZones {zid}: cost {r.get('cost_usd')} vs "
                          f"{cost_want:.2f} (area x catalogue $/sf)")
        deck_cost_total += cost_want
    missing = sorted(set(zones) - seen_zones)
    if missing:
        errors.append(f"DeckZones: missing zones {', '.join(missing)} "
                      f"(have {len(seen_zones)}, want {len(zones)})")
    distinct = {str(r.get("deck_code") or "").strip()
                for r in deck_rows if str(r.get("zone_id") or "").strip() in zones}
    if len(distinct) > 4:
        errors.append(f"DeckZones: {len(distinct)} distinct deck types, "
                      f"cap is 4")

    seen_lines, chord_cost_total = set(), 0.0
    for r in chord_rows:
        lid = str(r.get("chord_id") or "").strip()
        if not lid:
            continue
        seen_lines.add(lid)
        c = chords.get(lid)
        if c is None:
            errors.append(f"ChordSchedule: line {lid!r} is not an S-201 line "
                          f"(want: {', '.join(sorted(chords))})")
            continue
        if not _close(_num(r.get("length_ft")), float(c["length_ft"]), 0.02, 1.0):
            errors.append(f"ChordSchedule {lid}: length {r.get('length_ft')} "
                          f"vs {c['length_ft']}")
        code = str(r.get("section_code") or "").strip()
        cat = chord_cat.get(code)
        if cat is None:
            errors.append(f"ChordSchedule {lid}: section_code {code!r} not in "
                          f"the chord catalogue (valid: "
                          f"{', '.join(sorted(chord_cat))})")
            continue
        cost_want = (float(c["length_ft"]) * float(cat["weight_plf"])
                     * chord_per_lb)
        if not _close(_num(r.get("cost_usd")), cost_want, 0.01, 1.0):
            errors.append(f"ChordSchedule {lid}: cost {r.get('cost_usd')} vs "
                          f"{cost_want:.2f} (length x weight x ${chord_per_lb}/lb)")
        chord_cost_total += cost_want
    missing = sorted(set(chords) - seen_lines)
    if missing:
        errors.append(f"ChordSchedule: missing lines {', '.join(missing)} "
                      f"(have {len(seen_lines)}, want {len(chords)})")

    cost_rows = _sheet_from_zip(zip_path, "Cost")
    if cost_rows:
        for r in cost_rows:
            if re.match(r"^\s*(grand\s+)?(total|totals|sum|subtotal|sub-total|all)\b",
                        str(r.get("item") or ""), re.I):
                errors.append("Cost: delete the TOTAL row — the grader sums "
                              "every Cost row, so a total row double-counts; "
                              "the item rows alone must sum to the installed cost")
                break
        deck_sum = sum(_num(r.get("total_usd")) or 0.0 for r in cost_rows
                       if "deck" in str(r.get("item") or "").lower())
        chord_sum = sum(_num(r.get("total_usd")) or 0.0 for r in cost_rows
                        if "chord" in str(r.get("item") or "").lower()
                        or "collector" in str(r.get("item") or "").lower())
        if deck_sum and not _close(deck_sum, deck_cost_total, 0.01, 1.0):
            errors.append(f"Cost: deck subtotal {deck_sum:.2f} vs "
                          f"DeckZones sum {deck_cost_total:.2f}")
        if chord_sum and not _close(chord_sum, chord_cost_total, 0.01, 1.0):
            errors.append(f"Cost: chord/collector subtotal {chord_sum:.2f} vs "
                          f"ChordSchedule sum {chord_cost_total:.2f}")
    return errors


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--zones", required=True)
    ap.add_argument("--chords", required=True)
    ap.add_argument("--deck-catalogue", required=True)
    ap.add_argument("--chord-catalogue", required=True)
    ap.add_argument("--meta", default=None)
    args = ap.parse_args(argv)
    for p in (args.zip, args.zones, args.chords, args.deck_catalogue,
              args.chord_catalogue):
        if not os.path.exists(p):
            print(f"SCHEME UNAVAILABLE: missing input {p}")
            return 2
    try:
        errors = check(args.zip, args.zones, args.chords, args.deck_catalogue,
                       args.chord_catalogue, args.meta)
    except Exception as exc:  # noqa: BLE001 - fail-open: unavailable, not FAIL
        print(f"SCHEME UNAVAILABLE: {exc}")
        return 2
    if not errors:
        print("SCHEME PASS")
        return 0
    print("SCHEME FAIL")
    for e in errors:
        print("ERR " + e)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
