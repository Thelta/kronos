from __future__ import annotations

import json
import logging
import re
import urllib.request

from rapidfuzz import process as rfprocess

logger = logging.getLogger(__name__)

_FUZZY_MATCH_THRESHOLD = 70
_cached_student_names: list[str] | None = None


def load_student_names() -> list[str]:
    global _cached_student_names
    if _cached_student_names is not None:
        return _cached_student_names
    try:
        req = urllib.request.Request(
            "https://schaledb.com/data/jp/students.min.json",
            headers={"User-Agent": "kronos-analyzer"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        names = [entry["Name"] for entry in data.values() if "Name" in entry]
        _cached_student_names = names
        logger.info("Loaded %d student names from schaledb", len(names))
    except Exception:
        logger.warning("Failed to fetch student names from schaledb, falling back to raw names")
        _cached_student_names = []
    return _cached_student_names


def resolve_name(raw_name: str, known_names: list[str]) -> str:
    if not raw_name or not known_names:
        return raw_name
    result = rfprocess.extractOne(raw_name, known_names)
    if result is None:
        return raw_name
    match_name, score, _ = result
    if score >= _FUZZY_MATCH_THRESHOLD:
        return match_name
    return raw_name
