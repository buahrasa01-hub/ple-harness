#!/usr/bin/env python3
"""Robust IFC-SPF reader for the PLE harness (run 44842138147 fix).

Why this exists: in run 44842138147 the agent burned steps 23-65 hand-rolling
an IFC parser with regexes. Its ``args_of`` split on commas inside quoted
descriptions ("The description contains a comma, breaking arg parsing..."),
linked zero properties to the 63 panels it found, and the run died on the
daily quota with the geometry never extracted. The deliverable that got
graded therefore used invented uniform piers and a footprint from the wrong
drawings.

This is a *generic* STEP-file tokenizer + entity reader: it knows nothing
about this task's building, codes, or answers. It only parses the file
format correctly:

- quoted strings may contain commas, parentheses, and doubled-quote escapes;
- nested argument lists, ``$`` (omitted), ``*`` (any), ``#id`` references,
  ``.ENUM.`` values, ``.T.``/``.F.``, typed values like ``IFCREAL(0.0)``;
- entities may span several lines.

Usage:
    python3 ifc_extract.py model.ifc [--out panels.json]

Prints a JSON summary (walls with marks, properties, placements, profiles)
to stdout. Exit 0 when at least one wall was extracted, 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


class StepError(ValueError):
    """Raised when the STEP text cannot be tokenized or parsed."""


def tokenize(text: str):
    """Split STEP text into tokens.

    Tokens are tuples ``(kind, value)`` with kind in ``str, ref, num, enum,
    ident, punct``. Comments (``/* ... */``) are skipped.
    """
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                raise StepError("unterminated comment")
            i = end + 2
        elif c == "'":
            # Quoted string; '' inside is an escaped quote. Commas and
            # parens inside are data, not structure (the run-44842138147 bug).
            j = i + 1
            buf = []
            while j < n:
                if text[j] == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        buf.append("'")
                        j += 2
                    else:
                        break
                else:
                    buf.append(text[j])
                    j += 1
            if j >= n:
                raise StepError("unterminated string")
            tokens.append(("str", "".join(buf)))
            i = j + 1
        elif c == "#":
            j = i + 1
            while j < n and text[j].isdigit():
                j += 1
            if j == i + 1:
                raise StepError("'#' not followed by digits")
            tokens.append(("ref", int(text[i + 1:j])))
            i = j
        elif c == ".":
            j = text.find(".", i + 1)
            if j < 0:
                raise StepError("unterminated .ENUM.")
            tokens.append(("enum", text[i + 1:j]))
            i = j + 1
        elif c in "(),;=$*":
            tokens.append(("punct", c))
            i += 1
        elif c.isdigit() or (c in "+-" and i + 1 < n and (text[i + 1].isdigit() or text[i + 1] == ".")):
            j = i
            while j < n and text[j] in "+-.0123456789eE":
                j += 1
            raw = text[i:j]
            try:
                tokens.append(("num", float(raw) if any(k in raw for k in ".eE") else int(raw)))
            except ValueError as exc:
                raise StepError(f"bad number {raw!r}") from exc
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_-"):
                j += 1
            tokens.append(("ident", text[i:j]))
            i = j
        else:
            raise StepError(f"unexpected character {c!r} at offset {i}")
    return tokens


class Parser:
    """Recursive-descent parser over the token stream."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def next(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, kind, value=None):
        tok = self.next()
        if tok[0] != kind or (value is not None and tok[1] != value):
            raise StepError(f"expected {kind} {value!r}, got {tok!r}")
        return tok

    def parse_value(self):
        kind, value = self.peek()
        if kind == "punct" and value == "(":
            self.next()
            items = []
            if self.peek() == ("punct", ")"):
                self.next()
                return items
            while True:
                items.append(self.parse_value())
                k, v = self.next()
                if (k, v) == ("punct", ")"):
                    return items
                if (k, v) != ("punct", ","):
                    raise StepError(f"expected ',' or ')' in list, got {(k, v)!r}")
        if kind == "punct" and value == "$":
            self.next()
            return None
        if kind == "punct" and value == "*":
            self.next()
            return ("any",)
        if kind == "ref":
            self.next()
            return ("ref", value)
        if kind == "enum":
            self.next()
            return ("enum", value)
        if kind in ("str", "num"):
            self.next()
            return value
        if kind == "ident":
            # Typed value: TYPE(value) — e.g. IFCREAL(0.0), IFCLABEL('S01').
            name = value
            self.next()
            if self.peek() == ("punct", "("):
                self.next()
                inner = self.parse_value()
                self.expect("punct", ")")
                if name in ("T", "F"):
                    return name == "T"
                return ("typed", name, inner)
            if name == "T":
                return True
            if name == "F":
                return False
            return ("ident", name)
        raise StepError(f"unexpected token {kind} {value!r} at position {self.pos}")

    def parse_entities(self):
        """Return {entity_id: (type_name, [args])} for every #id=TYPE(...);."""
        entities = {}
        while self.pos < len(self.tokens):
            kind, value = self.peek()
            if kind == "ident" and value == "DATA":
                self.next()
                self.expect("punct", ";")
                continue
            if kind == "ident" and value in ("HEADER", "ENDSEC", "END-ISO-10303-21", "ISO-10303-21"):
                # Skip header section wholesale: consume to the next ';' at depth 0.
                depth = 0
                while self.pos < len(self.tokens):
                    k, v = self.next()
                    if (k, v) == ("punct", "("):
                        depth += 1
                    elif (k, v) == ("punct", ")"):
                        depth -= 1
                    elif (k, v) == ("punct", ";") and depth == 0:
                        break
                continue
            if kind == "ref":
                self.next()
                self.expect("punct", "=")
                tkind, tname = self.next()
                if tkind != "ident":
                    raise StepError(f"expected entity type, got {(tkind, tname)!r}")
                self.expect("punct", "(")
                args = []
                if self.peek() != ("punct", ")"):
                    while True:
                        args.append(self.parse_value())
                        k, v = self.next()
                        if (k, v) == ("punct", ")"):
                            break
                        if (k, v) != ("punct", ","):
                            raise StepError(f"expected ',' or ')' in entity, got {(k, v)!r}")
                else:
                    self.next()
                self.expect("punct", ";")
                entities[value] = (tname, args)
                continue
            # Anything else (section keywords, stray ';') is skipped.
            self.next()
        return entities


def _unwrap(value):
    """Unwrap ('typed', name, inner) / ('enum', name) to plain values."""
    if isinstance(value, tuple):
        if len(value) == 3 and value[0] == "typed":
            return _unwrap(value[2])
        if len(value) == 2 and value[0] == "enum":
            return value[1]
        if len(value) == 2 and value[0] == "ref":
            return value
        if len(value) == 1 and value[0] == "any":
            return None
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def _ref(value):
    return value[1] if isinstance(value, tuple) and len(value) == 2 and value[0] == "ref" else None


def extract_walls(entities):
    """Resolve walls, their property sets, placements and profiles."""
    # Property sets: id -> {name: value}
    prop_sets = {}
    for eid, (etype, args) in entities.items():
        if etype.upper() == "IFCPROPERTYSET" and len(args) >= 5:
            props = {}
            for pref in args[4] if isinstance(args[4], list) else []:
                pid = _ref(pref)
                if pid is None or pid not in entities:
                    continue
                ptype, pargs = entities[pid]
                if ptype.upper() == "IFCPROPERTYSINGLEVALUE" and len(pargs) >= 3:
                    props[_unwrap(pargs[0])] = _unwrap(pargs[2])
            prop_sets[eid] = props
    # Wall -> property sets via IfcRelDefinesByProperties.
    wall_props = {}
    for eid, (etype, args) in entities.items():
        if etype.upper() == "IFCRELDEFINESBYPROPERTIES" and len(args) >= 6:
            related = args[4] if isinstance(args[4], list) else [args[4]]
            psid = _ref(args[5])
            if psid is None or psid not in prop_sets:
                continue
            for rel in related:
                rid = _ref(rel)
                if rid is not None:
                    wall_props.setdefault(rid, {}).update(prop_sets[psid])
    # Placement chains: id -> (parent_id, [x, y, z]).
    placements = {}
    for eid, (etype, args) in entities.items():
        if etype.upper() == "IFCLOCALPLACEMENT" and len(args) >= 2:
            parent = _ref(args[0])
            loc = None
            rel = _ref(args[1])
            if rel is not None and rel in entities:
                rtype, rargs = entities[rel]
                if rtype.upper() == "IFCAXIS2PLACEMENT3D" and rargs:
                    pt = _ref(rargs[0])
                    if pt is not None and pt in entities:
                        _, pargs = entities[pt]
                        if pargs and isinstance(pargs[0], list):
                            coords = [_unwrap(c) for c in pargs[0]]
                            if len(coords) >= 3 and all(isinstance(c, (int, float)) for c in coords[:3]):
                                loc = [float(c) for c in coords[:3]]
            placements[eid] = (parent, loc)

    def global_point(placement_id):
        """Sum translations along the placement chain (rotations ignored).

        Documented limitation: rotation is not composed. For the panel
        set-out (positions along wall lines) the translations are what
        matters; if a model needs rotations, resolve them from the
        IfcAxis2Placement3D Axis/RefDirection entries.
        """
        total = [0.0, 0.0, 0.0]
        seen, cur = set(), placement_id
        while cur is not None and cur in placements and cur not in seen:
            seen.add(cur)
            parent, loc = placements[cur]
            if loc:
                total = [a + b for a, b in zip(total, loc)]
            cur = parent
        return total

    # Openings per wall via IfcRelVoidsElement.
    openings = {}
    for eid, (etype, args) in entities.items():
        if etype.upper() == "IFCRELVOIDSELEMENT" and len(args) >= 5:
            wid = _ref(args[4])
            if wid is not None:
                openings[wid] = openings.get(wid, 0) + 1

    walls = []
    for eid, (etype, args) in entities.items():
        if etype.upper() not in ("IFCWALL", "IFCWALLSTANDARDCASE"):
            continue
        name = _unwrap(args[2]) if len(args) > 2 else None
        tag = _unwrap(args[7]) if len(args) > 7 else None
        placement_id = _ref(args[5]) if len(args) > 5 else None
        props = wall_props.get(eid, {})
        profile = None
        rep = _ref(args[6]) if len(args) > 6 else None
        if rep is not None and rep in entities:
            _, rargs = entities[rep]
            # IfcProductDefinitionShape: (Name, Description, Representations).
            reps = rargs[2] if len(rargs) > 2 and isinstance(rargs[2], list) else []
            for item in reps:
                iid = _ref(item)
                if iid is None or iid not in entities:
                    continue
                itype, iargs = entities[iid]
                # A shape representation holds its solids in Items (arg 3).
                solids = []
                if itype.upper() == "IFCSHAPEREPRESENTATION" and len(iargs) > 3:
                    solids = iargs[3] if isinstance(iargs[3], list) else []
                elif itype.upper() == "IFCEXTRUDEDAREASOLID":
                    solids = [item]
                for solid in solids:
                    sid = _ref(solid)
                    if sid is None or sid not in entities:
                        continue
                    stype, sargs = entities[sid]
                    if stype.upper() != "IFCEXTRUDEDAREASOLID" or not sargs:
                        continue
                    swept = _ref(sargs[0])
                    if swept is not None and swept in entities:
                        ptype, pargs = entities[swept]
                        if ptype.upper() == "IFCRECTANGLEPROFILEDEF" and len(pargs) >= 5:
                            # (ProfileType, ProfileName, Position, XDim, YDim).
                            profile = {
                                "type": "rectangle",
                                "x_dim": _unwrap(pargs[3]),
                                "y_dim": _unwrap(pargs[4]),
                            }
                        else:
                            profile = {"type": ptype}
                    break
                if profile is not None:
                    break
        walls.append({
            "entity_id": eid,
            "global_id": _unwrap(args[0]) if args else None,
            "name": name,
            "mark": props.get("PanelMark", name),
            "tag": tag,
            "properties": props,
            "placement_xyz": global_point(placement_id) if placement_id else None,
            "profile": profile,
            "openings": openings.get(eid, 0),
        })
    walls.sort(key=lambda w: (str(w["mark"]), w["entity_id"]))
    return walls


def main(argv):
    if len(argv) < 2:
        print("usage: ifc_extract.py model.ifc [--out panels.json]", file=sys.stderr)
        return 2
    path = Path(argv[1])
    out = None
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2
    try:
        entities = Parser(tokenize(text)).parse_entities()
    except StepError as exc:
        print(f"STEP parse error: {exc}", file=sys.stderr)
        return 1
    walls = extract_walls(entities)
    result = {
        "file": str(path),
        "entities": len(entities),
        "n_walls": len(walls),
        "walls": walls,
    }
    payload = json.dumps(result, indent=1, default=str)
    if out:
        out.write_text(payload, encoding="utf-8")
    print(payload)
    if not walls:
        print("no IfcWall entities found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
