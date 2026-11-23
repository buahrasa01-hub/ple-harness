# zones_extract.py — extract the deck-zone / chord-line scheme from S-201.
#
# Run 44965326609 (22 Nov) scored 0.011111 with the same bottleneck as
# run 44842138147: the agent invented its own zone/chord IDs (D1/D2,
# C-N, COLL-*) instead of the scheme the task documents define. The
# scheme IS in the documents, on sheet S-201 of the roof framing plan:
# zone labels Z11..Z43 with their rectangles, chord/collector line
# labels and lines, and the notes ("ZONES Z11 THROUGH Z43 AS SHOWN",
# "ZONE Z22 ... DECK SPAN 4'-0\"", "ZONE Z43 ... GENERATOR YARD NOTCH").
# The agent never read the DXF text, so it never learned the scheme.
#
# This helper makes the scheme machine-readable at run start. It contains
# no answers: only the zone IDs + areas, the line IDs + lengths, and the
# two catalogues as printed in the task's own procurement files. The agent
# still does every bit of engineering (wind loads, shears, deck/chord
# selection, piers, register). The outputs are plain CSVs the agent and
# scheme_check.py both read.
#
# Usage:
#   python3 zones_extract.py --taskdir <task folder> --out <dir>
# Writes zones.csv, chords.csv, deck_catalogue.csv, chord_catalogue.csv,
# s201_notes.txt, meta.json into --out. Exit 0 on success, 1 on failure.
"""Extract deck zones, chord lines and catalogues from the task files."""

import argparse
import csv
import json
import math
import os
import re
import sys

ZONE_LABEL = re.compile(r"^Z\d\d$")
LINE_LABEL = re.compile(r"^[CKO]-[A-Z0-9]+$")
SPAN_NOTE = re.compile(r"Z(\d\d)[^\n]{0,120}?DECK SPAN\s*(\d+)", re.I)


def _shoelace(pts):
    area = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _point_in_poly(x, y, poly):
    """Ray-casting point-in-polygon (boundary counts as inside)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if min(y1, y2) < y <= max(y1, y2) and x <= max(x1, x2):
            xinters = x1 + (y - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
            if x <= xinters:
                inside = not inside
    return inside


def _poly_intersection_area(poly_a, poly_b):
    """Exact intersection area of two (orthogonal) polygons.

    Subdivides the plane on every distinct vertex coordinate of both
    polygons and sums the cells whose centre lies inside both. Exact
    whenever both polygons' edges are axis-aligned or diagonal-free
    between the shared coordinates, which holds for the S-201 geometry.
    """
    xs = sorted({round(p[0], 6) for p in poly_a} | {round(p[0], 6) for p in poly_b})
    ys = sorted({round(p[1], 6) for p in poly_a} | {round(p[1], 6) for p in poly_b})
    total = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2.0, (ys[j] + ys[j + 1]) / 2.0
            if _point_in_poly(cx, cy, poly_a) and _point_in_poly(cx, cy, poly_b):
                total += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return total


def _find_dxf(taskdir):
    for root, _dirs, files in os.walk(taskdir):
        for f in files:
            if re.search(r"S-201.*\.dxf$", f, re.I):
                return os.path.join(root, f)
    return None


def _find_xlsx(taskdir, header_want):
    """First .xlsx whose first sheet has a header cell == header_want."""
    import openpyxl

    for root, _dirs, files in os.walk(taskdir):
        for f in sorted(files):
            if not f.lower().endswith(".xlsx"):
                continue
            path = os.path.join(root, f)
            try:
                wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
            except Exception:
                continue
            for ws in wb.worksheets:
                try:
                    header = [str(c.value).strip().lower() if c.value is not None else ""
                              for c in next(ws.iter_rows(min_row=1, max_row=1))]
                except StopIteration:
                    continue
                if header_want.lower() in header:
                    return path
            wb.close()
    return None


def _read_table(path, key_col):
    """Read the first sheet whose header contains key_col; return rows + notes."""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    rows, notes = [], ""
    for ws in wb.worksheets:
        header = [str(c.value).strip() if c.value is not None else ""
                  for c in next(ws.iter_rows(min_row=1, max_row=1))]
        lower = [h.lower() for h in header]
        if key_col.lower() in lower:
            for r in ws.iter_rows(min_row=2, values_only=True):
                if all(v is None for v in r):
                    continue
                rows.append(dict(zip(header, r)))
        elif ws.title.lower().startswith("note"):
            texts = []
            for row in ws.iter_rows(min_row=2, values_only=True):
                if len(row) > 1 and row[1] is not None:
                    texts.append(str(row[1]))
            notes = "\n".join(texts)
    wb.close()
    return rows, notes


def extract(taskdir):
    import ezdxf

    dxf_path = _find_dxf(taskdir)
    if not dxf_path:
        raise RuntimeError("no S-201 DXF found under taskdir")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    labels = {}   # zone id -> (x, y)
    lines = {}    # line id -> (x, y)
    notes = []
    for e in msp:
        if e.dxftype() != "TEXT":
            continue
        text = (e.dxf.text or "").strip()
        x, y = float(e.dxf.insert.x), float(e.dxf.insert.y)
        if ZONE_LABEL.match(text):
            labels[text] = (x, y)
        elif LINE_LABEL.match(text):
            lines[text] = (x, y)
        elif len(text) > 25:
            notes.append(text)

    rects, cuts = [], []
    for e in msp:
        if e.dxf.layer != "S-DECK-ZONE" or e.dxftype() != "LWPOLYLINE":
            continue
        if not getattr(e, "closed", False):
            continue
        pts = [(float(p[0]), float(p[1])) for p in e.get_points()]
        if len(pts) < 3:
            continue
        if len(pts) == 4 and 3000.0 <= _shoelace(pts) <= 4000.0:
            rects.append(pts)
        else:
            cuts.append(pts)
    if not rects:
        raise RuntimeError("no 60x60 zone rectangles on S-DECK-ZONE")
    if not cuts:
        raise RuntimeError("no outline/opening polygons on S-DECK-ZONE")
    outline = max(cuts, key=_shoelace)
    openings = [c for c in cuts if c is not outline]

    spans = {}
    for note in notes:
        m = SPAN_NOTE.search(note)
        if m:
            spans["Z" + m.group(1)] = int(m.group(2))

    zones = []
    for label, (lx, ly) in sorted(labels.items()):
        rect = None
        for pts in rects:
            if _point_in_poly(lx, ly, pts):
                rect = pts
                break
        if rect is None:
            raise RuntimeError(f"zone label {label} sits in no rectangle")
        xs = [p[0] for p in rect]
        ys = [p[1] for p in rect]
        area = _poly_intersection_area(rect, outline)
        for op in openings:
            area -= _poly_intersection_area(rect, op)
        zones.append({
            "zone_id": label,
            "x0_ft": min(xs), "x1_ft": max(xs),
            "y0_ft": min(ys), "y1_ft": max(ys),
            "area_sf": round(area, 1),
            "span_ft": spans.get(label, 6),
        })
    if len(zones) != len(rects):
        raise RuntimeError(
            f"zone labels ({len(zones)}) != zone rectangles ({len(rects)})")

    # Chord / collector / opening lines: LINE entities on S-CHORD/S-COLL,
    # matched to the nearest label by midpoint distance.
    segments = []
    for e in msp:
        if e.dxf.layer not in ("S-CHORD", "S-COLL") or e.dxftype() != "LINE":
            continue
        s = (float(e.dxf.start.x), float(e.dxf.start.y))
        t = (float(e.dxf.end.x), float(e.dxf.end.y))
        segments.append((s, t))
    chords = []
    for label, (lx, ly) in sorted(lines.items()):
        best, best_d = None, None
        for s, t in segments:
            mx, my = (s[0] + t[0]) / 2.0, (s[1] + t[1]) / 2.0
            d = math.hypot(lx - mx, ly - my)
            if best_d is None or d < best_d:
                best, best_d = (s, t), d
        if best is None or best_d > 25.0:
            raise RuntimeError(f"line label {label} matches no segment")
        (x1, y1), (x2, y2) = best
        horizontal = abs(y2 - y1) < abs(x2 - x1)
        length = abs(x2 - x1) if horizontal else abs(y2 - y1)
        role = {"C": "chord", "K": "collector", "O": "opening"}[label[0]]
        chords.append({
            "line_id": label,
            "axis": "X" if horizontal else "Y",
            "s0_ft": round(min(x1, x2) if horizontal else min(y1, y2), 1),
            "s1_ft": round(max(x1, x2) if horizontal else max(y1, y2), 1),
            "length_ft": round(length, 1),
            "role": role,
        })

    deck_path = _find_xlsx(taskdir, "deck_code")
    chord_path = _find_xlsx(taskdir, "section_code")
    if not deck_path or not chord_path:
        raise RuntimeError("catalogue workbooks not found under taskdir")
    deck_rows, deck_notes = _read_table(deck_path, "deck_code")
    chord_rows, chord_notes = _read_table(chord_path, "section_code")

    omega = 2.35
    m = re.search(r"omega_d\s*=\s*(\d+(?:\.\d+)?)", deck_notes)
    if m:
        omega = float(m.group(1))
    span_factor = 1.15
    m = re.search(r"multiply the tabulated G'\s*by\s*(\d+(?:\.\d+)?)", deck_notes)
    if m:
        span_factor = float(m.group(1))
    chord_per_lb = 2.85
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)\s*per pound", chord_notes)
    if m:
        chord_per_lb = float(m.group(1))

    deck_cat = []
    for r in deck_rows:
        sn = float(r["nominal_shear_strength_plf"])
        deck_cat.append({
            "deck_code": r["deck_code"],
            "gauge": r["gauge"],
            "nominal_shear_strength_plf": sn,
            "shear_stiffness_G_prime_kip_per_in": float(r["shear_stiffness_G_prime_kip_per_in"]),
            "installed_cost_usd_per_sf": float(r["installed_cost_usd_per_sf"]),
            "asd_plf": round(sn / omega, 1),
        })
    chord_cat = []
    for r in chord_rows:
        chord_cat.append({
            "section_code": r["section_code"],
            "area_in2": float(r["area_in2"]),
            "weight_plf": float(r["weight_plf"]),
            "allowable_tension_kips": float(r["allowable_tension_kips"]),
            "allowable_compression_kips": float(r["allowable_compression_kips"]),
        })

    return {
        "zones": zones,
        "chords": chords,
        "deck_catalogue": deck_cat,
        "chord_catalogue": chord_cat,
        "notes": notes,
        "meta": {
            "omega_d": omega,
            "span_factor_4ft": span_factor,
            "chord_per_lb": chord_per_lb,
            "dxf": os.path.basename(dxf_path),
            "deck_workbook": os.path.basename(deck_path),
            "chord_workbook": os.path.basename(chord_path),
        },
    }


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r[f] for f in fields})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--taskdir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        data = extract(args.taskdir)
    except Exception as exc:  # noqa: BLE001 - fail-open: print and exit 1
        print(f"zones_extract: FAILED: {exc}", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    _write_csv(os.path.join(args.out, "zones.csv"), data["zones"],
               ["zone_id", "x0_ft", "x1_ft", "y0_ft", "y1_ft", "area_sf", "span_ft"])
    _write_csv(os.path.join(args.out, "chords.csv"), data["chords"],
               ["line_id", "axis", "s0_ft", "s1_ft", "length_ft", "role"])
    _write_csv(os.path.join(args.out, "deck_catalogue.csv"), data["deck_catalogue"],
               ["deck_code", "gauge", "nominal_shear_strength_plf",
                "shear_stiffness_G_prime_kip_per_in",
                "installed_cost_usd_per_sf", "asd_plf"])
    _write_csv(os.path.join(args.out, "chord_catalogue.csv"), data["chord_catalogue"],
               ["section_code", "area_in2", "weight_plf",
                "allowable_tension_kips", "allowable_compression_kips"])
    with open(os.path.join(args.out, "s201_notes.txt"), "w") as fh:
        fh.write("\n".join(data["notes"]) + "\n")
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(data["meta"], fh, indent=1)
    print(f"zones_extract: {len(data['zones'])} zones, "
          f"{len(data['chords'])} lines, "
          f"{len(data['deck_catalogue'])} deck codes, "
          f"{len(data['chord_catalogue'])} chord codes "
          f"from {data['meta']['dxf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
