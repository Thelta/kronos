"""Debug script for character_select name detection.

Usage:
    uv run python debug_character_select.py outputs/ocr/mpv-shot0001.json

Reads an OCR dump JSON, runs Lv-line-based slot detection and name detection,
and prints detailed info about what lands in each name region.
"""

import json
import sys

from kronos_analyzer.character_select import (
    CharacterSelectConfig,
    _is_text_candidate,
    box_center_in_region,
    box_intersects_region,
    bounds_from_box,
    build_slots_from_lv_lines,
    find_name_candidate_in_region,
    find_striker_anchor,
)
from kronos_analyzer.student_names import load_student_names, resolve_name
from kronos_analyzer.schemas import OCRDump, OCRLine


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python debug_character_select.py <ocr_dump.json> [image_width image_height]")
        sys.exit(1)

    dump_path = sys.argv[1]
    with open(dump_path, encoding="utf-8") as f:
        raw = json.load(f)

    lines = [OCRLine(text=l["text"], score=l.get("score"), box=l.get("box")) for l in raw["lines"]]
    dump = OCRDump(image=raw.get("image", ""), line_count=len(lines), combined_text="", lines=lines)

    # Infer image size from bounding boxes or accept as args
    if len(sys.argv) >= 4:
        img_w, img_h = int(sys.argv[2]), int(sys.argv[3])
    else:
        all_x = [p[0] for l in lines if l.box for p in l.box]
        all_y = [p[1] for l in lines if l.box for p in l.box]
        img_w = int(max(all_x) * 1.1) if all_x else 1920
        img_h = int(max(all_y) * 1.1) if all_y else 1080
    image_shape = (img_h, img_w, 3)
    print(f"Image shape: {img_w}x{img_h}")

    config = CharacterSelectConfig()

    # Find anchor
    anchor = find_striker_anchor(lines, image_shape, config)
    if anchor is None:
        print("STRIKER anchor NOT FOUND")
        sys.exit(1)

    print(f"Anchor: text={anchor.line.text!r} score={anchor.score:.1f} box={anchor.line.box}")
    anchor_left, anchor_top, anchor_right, anchor_bottom = bounds_from_box(anchor.line.box)
    anchor_width = anchor_right - anchor_left
    anchor_height = anchor_bottom - anchor_top
    print(f"Anchor bounds: left={anchor_left:.0f} top={anchor_top:.0f} right={anchor_right:.0f} bottom={anchor_bottom:.0f} w={anchor_width:.0f} h={anchor_height:.0f}")
    print()

    # Build slots from Lv lines
    slots = build_slots_from_lv_lines(lines, anchor.line, config)
    known_names = load_student_names()

    for slot in slots:
        print(f"=== Slot {slot.slot_index} [{slot.role}] ===")
        nr = slot.name_region
        lv_left, lv_top, lv_right, lv_bottom = bounds_from_box(slot.level_line.box)
        print(f"  lv_line:      {slot.level_line.text!r}  box=({lv_left:.0f},{lv_top:.0f},{lv_right:.0f},{lv_bottom:.0f})")
        print(f"  name_region:  ({nr[0]:.0f}, {nr[1]:.0f}, {nr[2]:.0f}, {nr[3]:.0f})")

        # Show ALL lines that intersect the name region
        print(f"  --- Lines near name region ---")
        for line in lines:
            if not line.box or not box_intersects_region(line.box, nr):
                continue
            left, top, right, bottom = bounds_from_box(line.box)
            is_candidate = _is_text_candidate(line, config.lv_pattern, min_length=1)
            center_ok = box_center_in_region(line.box, nr)
            if is_candidate and center_ok:
                tag = "PASS"
            elif is_candidate:
                tag = "EDGE"
            else:
                tag = "REJECT"
            print(f"    [{tag:6s}] {line.text!r:25s} score={line.score:.3f}  box=({left:.0f},{top:.0f},{right:.0f},{bottom:.0f})")

        # Show final result
        name_match = find_name_candidate_in_region(lines, nr, config.lv_pattern)
        raw_name = name_match.text if name_match else ""
        resolved = resolve_name(raw_name, known_names) if raw_name else ""
        if resolved and resolved != raw_name:
            print(f"  name:  {raw_name!r} -> {resolved!r}")
        else:
            print(f"  name:  {raw_name!r}")
        if name_match and name_match.secondary_line:
            print(f"    primary:   {name_match.primary_line.text!r}")
            print(f"    secondary: {name_match.secondary_line.text!r}")
        print()


if __name__ == "__main__":
    main()
