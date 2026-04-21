"""
rag/report_extractor.py
=======================

Flexible sleep report extraction pipeline.

Design philosophy:
  Extract everything first. Normalize what we recognize. Preserve what we don't.
  The LLM reasons on top of structured data — never a raw text dump.

Pipeline layers (all deterministic, no extra LLM calls):
  1. Broad extraction  — all label:value pairs, all sections, patient metadata
  2. Normalization     — alias mapping to canonical keys, value/unit parsing, flagging
  3. Patient grouping  — name-based grouping, ID fallback, unmatched bucket
  4. Trend analysis    — chronological comparison when ≥2 reports per patient
  5. Prompt formatting — structured text block for LLM injection

Public API:
    extract_report(text: str, filename: str) -> NormalizedReport
    group_by_patient(reports: list[NormalizedReport]) -> MultiReportContext
    format_multi_report_context(ctx: MultiReportContext) -> str
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedValue:
    """One data point extracted from a report."""
    numeric: Optional[float]  # None when value is non-numeric (name, date, etc.)
    unit: str                 # "%", "/hr", "min", "bpm", "" for non-numeric
    raw: str                  # original string as it appeared in the report
    flagged: bool = False
    flag_note: str = ""       # educational note — what this range is typically associated with


@dataclass
class NormalizedReport:
    """Fully processed single report."""
    filename: str
    is_sleep_report: bool
    report_type: str             # "PSG", "HST", "Actigraphy", "Sleep study"

    # Patient identity (all Optional — not all reports include all fields)
    patient_name: Optional[str]
    patient_id: Optional[str]
    patient_dob: Optional[str]
    patient_age: Optional[str]

    # Report metadata
    study_date: Optional[str]
    ordering_physician: Optional[str]
    facility: Optional[str]

    # Canonical metrics: key → ExtractedValue (e.g. "ahi" → ExtractedValue(22.1, "/hr", ...))
    metrics: dict[str, ExtractedValue] = field(default_factory=dict)

    # Everything that looked like structured data but didn't map to a canonical key
    other_findings: dict[str, str] = field(default_factory=dict)

    # Named text sections (impression, interpretation, recommendations, etc.)
    sections: dict[str, str] = field(default_factory=dict)

    # Raw report header (first ~600 chars) — patient name, dates, facility header
    raw_header: str = ""


@dataclass
class PatientGroup:
    """All reports belonging to one patient."""
    patient_key: str             # normalized key used for grouping (not for display)
    patient_name: Optional[str]  # best display name found across reports
    patient_id: Optional[str]
    reports: list[NormalizedReport] = field(default_factory=list)


@dataclass
class MultiReportContext:
    """Top-level object for a multi-report upload."""
    total_reports: int
    total_patients: int
    patient_groups: list[PatientGroup] = field(default_factory=list)
    unmatched_reports: list[NormalizedReport] = field(default_factory=list)  # no patient ID found


# =============================================================================
# SLEEP REPORT DETECTION
# =============================================================================

_SLEEP_KEYWORDS = [
    "polysomnography", "polysomnogram", "sleep study", "sleep efficiency",
    "sleep architecture", "ahi", "apnea-hypopnea", "apnea hypopnea",
    "hypopnea", "rera", "arousal index", "spo2", "oxygen saturation",
    "home sleep test", "actigraphy", "rem latency", "plm index",
    "obstructive apnea", "central apnea", "sleep onset latency",
    "wake after sleep onset", "waso", "total sleep time",
]


def _is_sleep_report(text: str) -> bool:
    lower = text.lower()
    return sum(1 for kw in _SLEEP_KEYWORDS if kw in lower) >= 2


def _detect_report_type(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["polysomnography", "polysomnogram", "in-lab psg", "attended psg"]):
        return "PSG (in-lab sleep study)"
    if any(k in lower for k in ["home sleep test", "hsat", "home sleep apnea test", "ambulatory"]):
        return "HST (home sleep test)"
    if "actigraphy" in lower:
        return "Actigraphy"
    return "Sleep study"


# =============================================================================
# METRIC ALIAS DICTIONARY
# Key: any label string that might appear in a real report (normalized to lowercase)
# Value: canonical metric key used in NormalizedReport.metrics
#
# This is the normalization layer — it is intentionally broad.
# Unknown labels are NOT discarded; they go to other_findings instead.
# =============================================================================

METRIC_ALIASES: dict[str, str] = {
    # ── Sleep continuity ─────────────────────────────────────────────────────
    "sleep efficiency": "sleep_efficiency",
    "sleep efficiency %": "sleep_efficiency",
    "sleep efficiency (%)": "sleep_efficiency",
    "se": "sleep_efficiency",

    "total sleep time": "total_sleep_time",
    "tst": "total_sleep_time",
    "total sleep time (min)": "total_sleep_time",
    "total sleep time, tst": "total_sleep_time",

    "sleep onset latency": "sleep_onset_latency",
    "sol": "sleep_onset_latency",
    "sleep latency": "sleep_onset_latency",
    "latency to sleep onset": "sleep_onset_latency",
    "latency to sleep": "sleep_onset_latency",
    "sleep onset": "sleep_onset_latency",

    "waso": "waso",
    "wake after sleep onset": "waso",
    "wake time after sleep onset": "waso",

    "total recording time": "total_recording_time",
    "trt": "total_recording_time",
    "time in bed": "time_in_bed",
    "tib": "time_in_bed",
    "time in bed, tib": "time_in_bed",

    # ── Sleep stages — percentage ─────────────────────────────────────────────
    "n1": "n1_pct",
    "stage n1": "n1_pct",
    "stage 1": "n1_pct",
    "nrem stage 1": "n1_pct",
    "n1 %": "n1_pct",

    "n2": "n2_pct",
    "stage n2": "n2_pct",
    "stage 2": "n2_pct",
    "nrem stage 2": "n2_pct",
    "n2 %": "n2_pct",

    "n3": "n3_pct",
    "stage n3": "n3_pct",
    "stage 3": "n3_pct",
    "nrem stage 3": "n3_pct",
    "deep sleep": "n3_pct",
    "sws": "n3_pct",
    "slow wave sleep": "n3_pct",
    "n3 %": "n3_pct",

    "rem": "rem_pct",
    "stage rem": "rem_pct",
    "stage r": "rem_pct",
    "rem sleep": "rem_pct",
    "r (rem)": "rem_pct",
    "rem %": "rem_pct",

    "rem latency": "rem_latency",
    "latency to rem": "rem_latency",
    "rem onset latency": "rem_latency",

    # ── Awakenings / arousal ──────────────────────────────────────────────────
    "awakenings": "awakenings",
    "number of awakenings": "awakenings",
    "total awakenings": "awakenings",
    "wake episodes": "awakenings",

    "arousal index": "arousal_index",
    "total arousal index": "arousal_index",
    "arousals per hour": "arousal_index",
    "arousals/hr": "arousal_index",
    "arousal index (per hour)": "arousal_index",

    "arousal count": "arousal_count",
    "total arousals": "arousal_count",
    "arousals": "arousal_count",

    # ── Breathing — respiratory events ───────────────────────────────────────
    "ahi": "ahi",
    "apnea hypopnea index": "ahi",
    "apnea-hypopnea index": "ahi",
    "overall ahi": "ahi",
    "total ahi": "ahi",
    "combined ahi": "ahi",
    "ahi (overall)": "ahi",
    "ahi (total)": "ahi",
    "ahi overall": "ahi",
    "ahi-total": "ahi",
    "ahi 3%": "ahi",
    "ahi 4%": "ahi",
    "apnea-hypopnea index 3%": "ahi",
    "apnea-hypopnea index 4%": "ahi",
    "apnea hypopnea index 3%": "ahi",
    "apnea hypopnea index 4%": "ahi",

    "rdi": "rdi",
    "respiratory disturbance index": "rdi",
    "total rdi": "rdi",
    "srdi": "rdi",

    "rera index": "rera_index",
    "rera": "rera_index",
    "respiratory effort related arousal index": "rera_index",

    "obstructive ahi": "obstructive_ahi",
    "obstructive apnea index": "obstructive_ahi",
    "oai": "obstructive_ahi",
    "sahi-obstructive": "obstructive_ahi",
    "sahi obstructive": "obstructive_ahi",

    "central ahi": "central_ahi",
    "central apnea index": "central_ahi",
    "cai": "central_ahi",
    "sahi-central": "central_ahi",
    "sahi central": "central_ahi",

    "mixed ahi": "mixed_ahi",
    "mixed apnea index": "mixed_ahi",

    "hypopnea index": "hypopnea_index",
    "hi": "hypopnea_index",

    "obstructive apneas": "obstructive_apneas",
    "central apneas": "central_apneas",
    "mixed apneas": "mixed_apneas",
    "hypopneas": "hypopneas",
    "total apneas": "total_apneas",
    "total respiratory events": "total_respiratory_events",

    # ── Oxygenation ───────────────────────────────────────────────────────────
    "spo2 minimum": "spo2_min",
    "minimum spo2": "spo2_min",
    "lowest spo2": "spo2_min",
    "spo2 nadir": "spo2_min",
    "o2 nadir": "spo2_min",
    "oxygen nadir": "spo2_min",
    "oxygen desaturation nadir": "spo2_min",
    "min spo2": "spo2_min",
    "spo2 min": "spo2_min",
    "spo2 min (%)": "spo2_min",

    "spo2 average": "spo2_avg",
    "average spo2": "spo2_avg",
    "mean spo2": "spo2_avg",
    "spo2 mean": "spo2_avg",
    "spo2 avg": "spo2_avg",
    "spo2 mean (%)": "spo2_avg",
    "spo2 max (%)": "spo2_max",
    "spo2 maximum": "spo2_max",
    "maximum spo2": "spo2_max",
    "highest spo2": "spo2_max",

    "time below 90": "spo2_below90_pct",
    "time below 90%": "spo2_below90_pct",
    "t90": "spo2_below90_pct",
    "ct90": "spo2_below90_pct",
    "time spo2 below 90": "spo2_below90_pct",
    "time spo2 < 90%": "spo2_below90_pct",
    "% time below 90%": "spo2_below90_pct",
    "oxygen desaturation time": "spo2_below90_pct",

    "oxygen desaturation index": "odi",
    "odi": "odi",
    "odi 3%": "odi",
    "odi 4%": "odi4",

    # ── Cardiac ───────────────────────────────────────────────────────────────
    "average heart rate": "hr_avg",
    "heart rate average": "hr_avg",
    "mean heart rate": "hr_avg",
    "hr average": "hr_avg",
    "hr mean": "hr_avg",
    "heart rate (avg)": "hr_avg",

    "minimum heart rate": "hr_min",
    "heart rate minimum": "hr_min",
    "hr minimum": "hr_min",
    "hr min": "hr_min",
    "lowest heart rate": "hr_min",

    "maximum heart rate": "hr_max",
    "heart rate maximum": "hr_max",
    "hr maximum": "hr_max",
    "hr max": "hr_max",
    "highest heart rate": "hr_max",

    # ── Movement ──────────────────────────────────────────────────────────────
    "plm index": "plm_index",
    "plmi": "plm_index",
    "periodic limb movement index": "plm_index",
    "plm": "plm_index",
    "plms/hr": "plm_index",
    "plm arousal index": "plm_arousal_index",
    "periodic limb movement arousal index": "plm_arousal_index",

    # ── Oxygenation burden ────────────────────────────────────────────────────
    "hypoxic burden": "hypoxic_burden",
    "hypoxic burden index": "hypoxic_burden",
    "oxygen burden": "hypoxic_burden",
    "oxygen desaturation burden": "hypoxic_burden",
    "desaturation burden": "hypoxic_burden",
}


# =============================================================================
# REFERENCE RANGES
# (low_thresh, high_thresh, note_if_below, note_if_above)
# None on a bound means no limit in that direction.
# =============================================================================

_RANGES: dict[str, tuple] = {
    "sleep_efficiency":    (85,   None, "below typical range (≥85%)",          None),
    "sleep_onset_latency": (None, 30,   None,                                   "above typical range (<20 min)"),
    "waso":                (None, 45,   None,                                   "above typical range (<30 min)"),
    "rem_pct":             (15,   30,   "below typical range (20–25%)",         "above typical range (20–25%)"),
    "n3_pct":              (10,   None, "below typical range (13–23%)",         None),
    "ahi":                 (None, 5,    None,                                   "above typical threshold (<5/hr)"),
    "rdi":                 (None, 5,    None,                                   "above typical threshold (<5/hr)"),
    "obstructive_ahi":     (None, 5,    None,                                   "above typical threshold (<5/hr)"),
    "rera_index":          (None, 5,    None,                                   "above typical threshold (<5/hr)"),
    "spo2_min":            (90,   None, "below typical threshold (≥90%)",       None),
    "spo2_avg":            (95,   None, "below typical range (≥95%)",           None),
    "spo2_below90_pct":    (None, 5,    None,                                   "above typical range (<5% of sleep time)"),
    "arousal_index":       (None, 15,   None,                                   "above typical range (<15/hr)"),
    "plm_index":           (None, 15,   None,                                   "above typical threshold (<15/hr)"),
    "odi":                 (None, 5,    None,                                   "above typical threshold (<5/hr)"),
    "hypoxic_burden":      (None, 32,   None,                                   "above typical threshold (<32 %min/h); higher values are associated with greater cardiovascular strain during sleep"),
}

# Direction that indicates improvement for trend analysis
_BETTER_LOWER = {"ahi", "rdi", "obstructive_ahi", "central_ahi", "mixed_ahi",
                 "rera_index", "arousal_index", "arousal_count", "plm_index",
                 "plm_arousal_index", "odi", "odi4", "spo2_below90_pct",
                 "sleep_onset_latency", "waso", "awakenings", "hypopnea_index",
                 "total_apneas", "total_respiratory_events", "hypoxic_burden"}
_BETTER_HIGHER = {"sleep_efficiency", "spo2_min", "spo2_avg", "n3_pct", "rem_pct"}


def _flag(key: str, value: float) -> tuple[bool, str]:
    if key not in _RANGES:
        return False, ""
    low, high, below_note, above_note = _RANGES[key]
    if low is not None and value <= low:
        return True, below_note or ""
    if high is not None and value > high:
        return True, above_note or ""
    return False, ""


# =============================================================================
# LAYER 1: BROAD EXTRACTION
# =============================================================================

# Matches "Label: Value" or "Label = Value" on the same line.
# Label: starts with a letter, up to 80 chars, can include spaces/hyphens/slashes/parens/commas.
# Value: rest of line, up to 140 chars.
_LABEL_VALUE_RE = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9 \-/\(\)\%,]{1,79}?)[ \t]*[:=][ \t]*([^\n]{1,140}?)[ \t]*$",
    re.MULTILINE,
)

# Matches "Label:\n<numeric value>" — label alone on a line, value on the next line.
# Value must start with a digit to avoid accidentally matching section headers as values.
_LABEL_VALUE_MULTILINE_RE = re.compile(
    r"^[ \t]*([A-Za-z][A-Za-z0-9 \-/\(\)\%,]{1,79}?)[ \t]*:[ \t]*\n[ \t]*(\d[^\n]*?)[ \t]*$",
    re.MULTILINE,
)

# Patient metadata — targeted patterns for common field names
_PATIENT_NAME_RE = re.compile(
    r"(?:patient(?:\s+name)?|subject|client|name)\s*[:=]\s*(.+?)(?:\s*\n|$)",
    re.IGNORECASE,
)
_PATIENT_ID_RE = re.compile(
    r"(?:patient\s+(?:id|number|#|no\.?)|mrn|medical\s+record\s+(?:number|no\.?)|pid|pt\.?\s*id)\s*[:=]\s*(.+?)(?:\s*\n|$)",
    re.IGNORECASE,
)
_DOB_RE = re.compile(
    r"(?:date\s+of\s+birth|dob|birth\s+date)\s*[:=]\s*(.+?)(?:\s*\n|$)",
    re.IGNORECASE,
)
_AGE_RE = re.compile(
    r"(?:age)\s*[:=]\s*(\d+(?:\s*(?:years?|yrs?|y/?o))?)",
    re.IGNORECASE,
)
_STUDY_DATE_RE = re.compile(
    r"(?:study\s+date|date\s+of\s+study|recording\s+date|test\s+date|date)\s*[:=]\s*(.+?)(?:\s*\n|$)",
    re.IGNORECASE,
)
_PHYSICIAN_RE = re.compile(
    r"(?:ordering\s+physician|physician|referring\s+physician|interpreted\s+by|read\s+by|dr\.?)\s*[:=]\s*(.+?)(?:\s*\n|$)",
    re.IGNORECASE,
)
_FACILITY_RE = re.compile(
    r"(?:facility|location|lab|clinic|hospital|center|centre)\s*[:=]\s*(.+?)(?:\s*\n|$)",
    re.IGNORECASE,
)


def _first_match(pattern: re.Pattern, text: str) -> Optional[str]:
    """Return the first capturing group from a pattern, stripped, or None."""
    m = pattern.search(text)
    if m:
        val = m.group(1).strip()
        return val if val else None
    return None


def _extract_patient_metadata(text: str) -> dict[str, Optional[str]]:
    """Extract patient identity and report metadata using targeted patterns."""
    # Only search the first 1200 chars — metadata is always at the top
    header = text[:1200]
    return {
        "patient_name":       _first_match(_PATIENT_NAME_RE, header),
        "patient_id":         _first_match(_PATIENT_ID_RE, header),
        "patient_dob":        _first_match(_DOB_RE, header),
        "patient_age":        _first_match(_AGE_RE, header),
        "study_date":         _first_match(_STUDY_DATE_RE, header),
        "ordering_physician": _first_match(_PHYSICIAN_RE, header),
        "facility":           _first_match(_FACILITY_RE, header),
    }


def _extract_all_label_values(text: str) -> dict[str, str]:
    """
    Broadly extract all "Label: Value" pairs in the text.
    Returns raw strings — normalization happens in the next layer.
    Deduplicates by keeping the first occurrence of each label.

    Two passes:
      1. Same-line  — "Label: Value"
      2. Multi-line — "Label:\\nValue" (value on the next line, must start with a digit)
    """
    found: dict[str, str] = {}

    # Pass 1: same-line format
    for m in _LABEL_VALUE_RE.finditer(text):
        label = re.sub(r"\s+", " ", m.group(1).strip())
        value = m.group(2).strip()
        if not value or value.endswith(":"):
            continue
        key = label.lower()
        if key not in found:
            found[key] = value

    # Pass 2: multi-line format (label on one line, numeric value on the next)
    for m in _LABEL_VALUE_MULTILINE_RE.finditer(text):
        label = re.sub(r"\s+", " ", m.group(1).strip())
        value = m.group(2).strip()
        if not value:
            continue
        key = label.lower()
        if key not in found:           # same-line takes precedence
            found[key] = value

    return found


def _extract_sections(text: str) -> dict[str, str]:
    """
    Extract named text sections from the report.
    A section header is a line of ≥3 uppercase characters (with optional spaces/hyphens),
    optionally ending with a colon. Body text runs until the next header or end of text.
    Caps each section at 900 chars.
    """
    # Header: ≥3 uppercase chars, can include spaces and hyphens, optional trailing colon
    header_re = re.compile(r"^([A-Z][A-Z \-/]{2,}):?\s*$", re.MULTILINE)
    headers = list(header_re.finditer(text))

    sections: dict[str, str] = {}
    for i, match in enumerate(headers):
        name = match.group(1).strip().lower()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        if body:
            # Truncate cleanly at sentence boundary
            if len(body) > 900:
                truncated = body[:900].rsplit(".", 1)
                body = (truncated[0] + ".") if len(truncated) > 1 else body[:900]
            sections[name] = body

    return sections


# =============================================================================
# LAYER 2: NORMALIZATION
# =============================================================================

_NUMERIC_RE = re.compile(r"(\d+(?:\.\d+)?)")
_UNIT_RE = re.compile(
    r"(%|events?/h(?:r(?:our)?)?|/h(?:r(?:our)?)?|bpm|beats?/min|min(?:utes?)?|"
    r"hours?|hrs?|l/min|times?|x|count)",
    re.IGNORECASE,
)

# Canonical keys that are percentages (unit should always be %)
_PCT_KEYS = {"sleep_efficiency", "n1_pct", "n2_pct", "n3_pct", "rem_pct",
             "spo2_min", "spo2_avg", "spo2_below90_pct"}
# Canonical keys whose value is in minutes
_MIN_KEYS = {"total_sleep_time", "sleep_onset_latency", "waso", "rem_latency",
             "total_recording_time", "time_in_bed"}


def _parse_value(raw: str, canonical_key: str = "") -> tuple[Optional[float], str]:
    """
    Extract a numeric value and unit from a raw value string.
    Uses canonical_key to infer expected unit when not explicit.
    Returns (None, "") if no numeric value is found.
    """
    # Handle "6h 32m" / "6:32" TST formats
    if canonical_key == "total_sleep_time":
        m = re.search(r"(\d+)\s*h(?:ours?)?\s*(\d+)\s*m(?:in)?", raw, re.IGNORECASE)
        if m:
            return float(int(m.group(1)) * 60 + int(m.group(2))), "min"
        m = re.search(r"(\d+):(\d{2})", raw)
        if m:
            return float(int(m.group(1)) * 60 + int(m.group(2))), "min"

    num_m = _NUMERIC_RE.search(raw)
    if not num_m:
        return None, ""

    value = float(num_m.group(1))
    unit_m = _UNIT_RE.search(raw)
    unit = unit_m.group(0) if unit_m else ""

    # Infer unit from canonical key when not found in raw string
    if not unit:
        if canonical_key in _PCT_KEYS or "%" in raw:
            unit = "%"
        elif canonical_key in _MIN_KEYS:
            unit = "min"
        elif canonical_key in {"ahi", "rdi", "arousal_index", "obstructive_ahi",
                                "central_ahi", "rera_index", "plm_index", "odi"}:
            unit = "/hr"

    unit = unit.strip()

    # Convert hours → minutes for time fields (e.g. "7h", "6 hrs", "7.5 hours")
    if canonical_key in _MIN_KEYS and re.match(r"^h(?:r(?:s|ours?)?)?$", unit, re.IGNORECASE):
        value = value * 60
        unit = "min"

    # Normalize unit variants: "events/hr", "event/hr" → "/hr"
    if re.search(r"events?/h", unit, re.IGNORECASE):
        unit = "/hr"

    return value, unit


def _normalize_label(label: str) -> Optional[str]:
    """
    Map a raw label string to a canonical metric key.
    Returns None if the label is not in the alias dictionary.
    """
    # Normalize: lowercase, collapse whitespace, strip
    normalized = re.sub(r"\s+", " ", label.lower().strip())
    return METRIC_ALIASES.get(normalized)


def _normalize_findings(
    raw_findings: dict[str, str]
) -> tuple[dict[str, ExtractedValue], dict[str, str]]:
    """
    Split raw_findings into:
      - metrics: known canonical keys with parsed values
      - other_findings: everything else, preserved as raw strings
    """
    metrics: dict[str, ExtractedValue] = {}
    other_findings: dict[str, str] = {}

    for label, raw_val in raw_findings.items():
        canonical = _normalize_label(label)
        if canonical:
            if canonical in metrics:
                # Already captured — skip duplicate (keep first/best)
                continue
            numeric, unit = _parse_value(raw_val, canonical)
            flagged, flag_note = _flag(canonical, numeric) if numeric is not None else (False, "")
            metrics[canonical] = ExtractedValue(
                numeric=numeric,
                unit=unit,
                raw=raw_val,
                flagged=flagged,
                flag_note=flag_note,
            )
        else:
            # Unknown label — preserve verbatim so nothing is lost
            # Filter out obvious non-data: very short labels, pure sentence fragments
            if len(label) >= 3 and not label.endswith(" and") and not label.endswith(" or"):
                other_findings[label] = raw_val

    return metrics, other_findings


# =============================================================================
# LAYER 1+2 COMBINED: MAIN EXTRACTOR
# =============================================================================

def extract_report(text: str, filename: str) -> NormalizedReport:
    """
    Extract and normalize all useful information from a single PDF's raw text.

    Returns a NormalizedReport with:
      - patient/report metadata
      - canonical metrics (normalized, flagged)
      - other_findings (anything structured but unrecognized)
      - sections (named text blocks)
      - raw_header (first 600 chars for context)

    If the text is not a sleep report, is_sleep_report=False and the caller
    should fall back to raw-dump handling.
    """
    meta = _extract_patient_metadata(text)
    raw_findings = _extract_all_label_values(text)
    sections = _extract_sections(text)
    metrics, other_findings = _normalize_findings(raw_findings)

    # Remove section content that leaked into label-value extraction
    # (ALL-CAPS section headers sometimes appear as label:value pairs with empty values)
    other_findings = {k: v for k, v in other_findings.items() if len(v.strip()) > 0}

    return NormalizedReport(
        filename=filename,
        is_sleep_report=_is_sleep_report(text),
        report_type=_detect_report_type(text),
        patient_name=meta["patient_name"],
        patient_id=meta["patient_id"],
        patient_dob=meta["patient_dob"],
        patient_age=meta["patient_age"],
        study_date=meta["study_date"],
        ordering_physician=meta["ordering_physician"],
        facility=meta["facility"],
        metrics=metrics,
        other_findings=other_findings,
        sections=sections,
        raw_header=text[:600].strip(),
    )


# =============================================================================
# LAYER 3: PATIENT GROUPING
# =============================================================================

def _normalize_patient_name(name: str) -> str:
    """
    Produce a stable, lowercase key from a patient name.
    Handles "Last, First" and "First Last" formats.
    """
    name = re.sub(r"\s+", " ", name.lower().strip())
    # Convert "Doe, John" → "john doe" for consistent matching
    if "," in name:
        parts = name.split(",", 1)
        last = parts[0].strip()
        first_word = parts[1].strip().split()[0] if parts[1].strip() else ""
        name = f"{first_word} {last}".strip()
    return name


def _patient_key(report: NormalizedReport) -> str:
    """Return a stable grouping key for a report's patient."""
    if report.patient_name:
        return "name:" + _normalize_patient_name(report.patient_name)
    if report.patient_id:
        return "id:" + report.patient_id.lower().strip()
    return f"file:{report.filename}"


def _sortable_date(date_str: Optional[str]) -> str:
    """Convert common date formats to YYYY-MM-DD for chronological sorting."""
    if not date_str:
        return ""
    # Already ISO
    if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
        return date_str
    # MM/DD/YYYY
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", date_str)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    # DD/MM/YYYY (ambiguous, but try)
    m = re.match(r"(\d{2})-(\d{2})-(\d{4})", date_str)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return date_str  # return as-is if no format matched


def group_by_patient(reports: list[NormalizedReport]) -> MultiReportContext:
    """
    Group a list of NormalizedReports by patient.

    Grouping priority: patient name → patient ID → filename (ungrouped).
    Reports within a group are sorted chronologically by study date.
    """
    groups: dict[str, PatientGroup] = {}
    unmatched: list[NormalizedReport] = []

    for report in reports:
        key = _patient_key(report)
        if key.startswith("file:"):
            unmatched.append(report)
            continue

        if key not in groups:
            groups[key] = PatientGroup(
                patient_key=key,
                patient_name=report.patient_name,
                patient_id=report.patient_id,
            )
        else:
            # Fill in missing identity fields if a later report has them
            g = groups[key]
            if not g.patient_name and report.patient_name:
                g.patient_name = report.patient_name
            if not g.patient_id and report.patient_id:
                g.patient_id = report.patient_id

        groups[key].reports.append(report)

    # Sort each patient's reports chronologically
    for group in groups.values():
        group.reports.sort(key=lambda r: _sortable_date(r.study_date))

    return MultiReportContext(
        total_reports=len(reports) + len(unmatched),
        total_patients=len(groups) + (1 if unmatched else 0),
        patient_groups=list(groups.values()),
        unmatched_reports=unmatched,
    )


# =============================================================================
# LAYER 4: TREND ANALYSIS
# =============================================================================

def _build_trends(group: PatientGroup) -> list[str]:
    """
    Compare metrics across a patient's reports (chronological order).
    Only compares metrics present in the FIRST and LAST report.
    Returns a list of formatted trend strings.
    """
    if len(group.reports) < 2:
        return []

    first = group.reports[0]
    last = group.reports[-1]

    # Find metrics with numeric values in both reports
    shared_keys = [
        k for k in first.metrics
        if k in last.metrics
        and first.metrics[k].numeric is not None
        and last.metrics[k].numeric is not None
    ]

    if not shared_keys:
        return []

    lines = []
    for key in shared_keys:
        v1 = first.metrics[key]
        v2 = last.metrics[key]
        delta = v2.numeric - v1.numeric  # type: ignore[operator]
        unit = v2.unit or v1.unit

        # Determine direction
        if key in _BETTER_LOWER:
            direction = "improved" if delta < 0 else ("worsened" if delta > 0 else "unchanged")
        elif key in _BETTER_HIGHER:
            direction = "improved" if delta > 0 else ("worsened" if delta < 0 else "unchanged")
        else:
            direction = ""

        label = key.replace("_", " ")
        delta_str = f"{delta:+.1f}" if delta != 0 else "no change"
        line = f"  {label}: {v1.numeric:.1f} → {v2.numeric:.1f} {unit}  ({delta_str}"
        if direction:
            line += f", {direction}"
        line += ")"
        lines.append(line)

    return lines


# =============================================================================
# LAYER 5: PROMPT FORMATTING
# =============================================================================

_DIVIDER = "━" * 48


def _fmt_metric(key: str, val: ExtractedValue) -> str:
    """Format a single metric line with optional NOTABLE flag."""
    label = key.replace("_", " ").title()
    if val.numeric is not None:
        num_str = f"{int(val.numeric)}" if val.numeric == int(val.numeric) else f"{val.numeric:.1f}"
        line = f"  {label}: {num_str} {val.unit}".rstrip()
    else:
        line = f"  {label}: {val.raw}"
    if val.flagged:
        line += f"  [NOTABLE: {val.flag_note}]"
    return line


def _fmt_report(report: NormalizedReport, index: int, total: int) -> list[str]:
    """Format a single report within a patient group."""
    lines = []

    # Report header line
    date_str = f"  Date: {report.study_date}" if report.study_date else ""
    facility_str = f"  Facility: {report.facility}" if report.facility else ""
    physician_str = f"  Physician: {report.ordering_physician}" if report.ordering_physician else ""
    meta_parts = [p for p in [date_str, facility_str, physician_str] if p]

    lines.append(f"\nREPORT {index} of {total}: {report.filename}")
    lines.append(f"Type: {report.report_type}")
    if meta_parts:
        lines.append(" | ".join(meta_parts).lstrip())

    # Metrics — prioritized canonical keys first, then the rest
    PRIORITY_KEYS = [
        "sleep_efficiency", "total_sleep_time", "sleep_onset_latency", "waso",
        "n1_pct", "n2_pct", "n3_pct", "rem_pct", "rem_latency",
        "awakenings", "arousal_index",
        "ahi", "rdi", "obstructive_ahi", "central_ahi", "rera_index",
        "spo2_min", "spo2_avg", "spo2_below90_pct", "odi",
        "hr_avg", "hr_min", "hr_max",
        "plm_index",
    ]

    if report.metrics:
        lines.append("\nKEY METRICS:")
        # Show priority keys first (if present), then remaining
        shown = set()
        for k in PRIORITY_KEYS:
            if k in report.metrics:
                lines.append(_fmt_metric(k, report.metrics[k]))
                shown.add(k)
        for k, v in report.metrics.items():
            if k not in shown:
                lines.append(_fmt_metric(k, v))
    else:
        lines.append("\nKEY METRICS: none extracted automatically")

    # Other findings (unrecognized but structured data)
    if report.other_findings:
        lines.append("\nOTHER FINDINGS:")
        for label, raw_val in report.other_findings.items():
            lines.append(f"  {label}: {raw_val}")

    # Clinical sections
    IMPORTANT_SECTIONS = ["impression", "clinical impression", "interpretation",
                          "summary", "recommendations", "recommendation",
                          "conclusion", "clinical summary", "study impression"]
    for sec_name in IMPORTANT_SECTIONS:
        if sec_name in report.sections:
            lines.append(f"\n{sec_name.upper()} (from report):")
            lines.append(report.sections[sec_name])

    # Any remaining sections not in the priority list
    shown_sections = set(IMPORTANT_SECTIONS)
    for sec_name, body in report.sections.items():
        if sec_name not in shown_sections:
            lines.append(f"\n{sec_name.upper()}:")
            lines.append(body)

    return lines


def format_multi_report_context(ctx: MultiReportContext) -> str:
    """
    Convert a MultiReportContext into a structured text block for LLM injection.

    Structure:
      - Summary: N reports, M patients
      - Per patient: identity header, then each report, then trends (if ≥2 reports)
      - Unmatched reports appended at the end
    """
    patient_word = "patient" if ctx.total_patients == 1 else "patients"
    report_word = "report" if ctx.total_reports == 1 else "reports"

    lines = [
        f"\n\n=== UPLOADED SLEEP REPORTS ===",
        f"Summary: {ctx.total_reports} {report_word} uploaded, "
        f"{ctx.total_patients} unique {patient_word} identified.",
    ]

    def _add_patient_group(group: PatientGroup):
        display_name = group.patient_name or group.patient_id or "Unknown Patient"
        n = len(group.reports)
        lines.append(f"\n{_DIVIDER}")
        lines.append(f"PATIENT: {display_name}  ·  {n} {'report' if n == 1 else 'reports'}")
        if group.patient_dob if hasattr(group, 'patient_dob') else False:
            pass  # DOB lives on reports, not group
        # Show DOB/age from first report if available
        r0 = group.reports[0]
        extras = []
        if r0.patient_dob:
            extras.append(f"DOB: {r0.patient_dob}")
        if r0.patient_age:
            extras.append(f"Age: {r0.patient_age}")
        if r0.patient_id:
            extras.append(f"ID: {r0.patient_id}")
        if extras:
            lines.append("  " + "  |  ".join(extras))
        lines.append(_DIVIDER)

        for i, report in enumerate(group.reports, 1):
            lines.extend(_fmt_report(report, i, n))

        trends = _build_trends(group)
        if trends:
            dates = [r.study_date for r in group.reports if r.study_date]
            date_range = f"{dates[0]} → {dates[-1]}" if len(dates) >= 2 else ""
            lines.append(f"\nTRENDS ({display_name}{', ' + date_range if date_range else ''}):")
            lines.extend(trends)

    for group in ctx.patient_groups:
        _add_patient_group(group)

    # Reports where no patient was identified
    if ctx.unmatched_reports:
        lines.append(f"\n{_DIVIDER}")
        lines.append(f"REPORTS WITHOUT IDENTIFIED PATIENT  ·  {len(ctx.unmatched_reports)} report(s)")
        lines.append(_DIVIDER)
        for i, report in enumerate(ctx.unmatched_reports, 1):
            lines.extend(_fmt_report(report, i, len(ctx.unmatched_reports)))

    lines.append("\n=== END SLEEP REPORTS ===")
    return "\n".join(lines)


# =============================================================================
# LLM FALLBACK + AUTO-LEARNING
# =============================================================================

from pathlib import Path as _Path
import json as _json

_LEARNED_ALIASES_PATH = _Path(__file__).parent.parent / "rag_artifacts" / "learned_aliases.json"


def _load_learned_aliases() -> None:
    """Load persisted learned aliases and merge into METRIC_ALIASES on import."""
    try:
        if _LEARNED_ALIASES_PATH.exists():
            with open(_LEARNED_ALIASES_PATH) as f:
                for label, canonical in _json.load(f).items():
                    if label not in METRIC_ALIASES:
                        METRIC_ALIASES[label] = canonical
    except Exception:
        pass


_load_learned_aliases()


def _reverse_map_label(text: str, value: float) -> Optional[str]:
    """
    Find the unique label that precedes `value` in the text.
    Returns the normalized label string, or None if ambiguous or not found.
    Only persists mappings that appear exactly once — avoids false associations.
    """
    val_strs = {str(value)}
    if value == int(value):
        val_strs.add(str(int(value)))
    val_strs.add(f"{value:.1f}")

    found: list[str] = []
    for val_str in val_strs:
        esc = re.escape(val_str)
        # Same-line: "Label: 92"
        for m in re.finditer(
            rf"([A-Za-z][A-Za-z0-9 \-/\(\)\%,]{{2,79}}?)\s*[:=]\s*{esc}\b",
            text, re.MULTILINE,
        ):
            label = re.sub(r"\s+", " ", m.group(1).strip()).lower()
            if label not in found:
                found.append(label)
        # Next-line: "Label:\n92"
        for m in re.finditer(
            rf"([A-Za-z][A-Za-z0-9 \-/\(\)\%,]{{2,79}}?)\s*:\s*\n\s*{esc}\b",
            text, re.MULTILINE,
        ):
            label = re.sub(r"\s+", " ", m.group(1).strip()).lower()
            if label not in found:
                found.append(label)

    return found[0] if len(found) == 1 else None


def _persist_learned_alias(label: str, canonical_key: str) -> None:
    """Save a new label→canonical_key mapping to disk and update in-memory METRIC_ALIASES."""
    try:
        existing: dict = {}
        if _LEARNED_ALIASES_PATH.exists():
            with open(_LEARNED_ALIASES_PATH) as f:
                existing = _json.load(f)
        if label not in existing and label not in METRIC_ALIASES:
            existing[label] = canonical_key
            _LEARNED_ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LEARNED_ALIASES_PATH, "w") as f:
                _json.dump(existing, f, indent=2, sort_keys=True)
            METRIC_ALIASES[label] = canonical_key
    except Exception:
        pass  # persistence is best-effort


_LLM_EXTRACTION_PROMPT = """\
Extract sleep study metrics from the report text below.
Return ONLY a valid JSON object. Include only metrics you can clearly identify — never guess.
Values must be plain numbers only (no units, no text).

Use exactly these key names (omit any you cannot find):
  ahi, rdi, obstructive_ahi, central_ahi, odi, arousal_index,
  sleep_efficiency, spo2_min, spo2_avg, spo2_below90_pct,
  total_sleep_time, time_in_bed, waso, sleep_onset_latency, rem_latency,
  n1_pct, n2_pct, n3_pct, rem_pct, plm_index, hypoxic_burden

Rules:
- sleep_efficiency and *_pct keys: 0–100 scale (e.g. 82, not 0.82)
- total_sleep_time, time_in_bed, waso, sleep_onset_latency, rem_latency: minutes
- All other keys: events/hr or native unit as it appears in the report

Report text:
"""


async def enrich_with_llm_fallback(
    report: "NormalizedReport",
    text: str,
    async_client,
) -> None:
    """
    If the report has fewer than 2 numeric metrics, call gpt-4o-mini to extract
    metrics from the PDF text and merge them into report.metrics.
    Auto-learns new label→canonical_key mappings from successful extractions.
    Mutates report in place. Never raises.
    """
    ahi_found = "ahi" in report.metrics and report.metrics["ahi"].numeric is not None
    total_metrics = sum(1 for v in report.metrics.values() if v.numeric is not None)
    if ahi_found and total_metrics >= 4:
        return

    try:
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _LLM_EXTRACTION_PROMPT + text[:6000]}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
            timeout=15,
        )
        extracted: dict = _json.loads(response.choices[0].message.content)

        for canonical_key, raw_val in extracted.items():
            if canonical_key in report.metrics:
                continue  # don't overwrite regex result
            try:
                numeric = float(raw_val)
            except (TypeError, ValueError):
                continue

            # LLM sometimes returns hours for TST/TIB despite instructions — convert to minutes.
            # Only for fields that are always hours-scale (TST/TIB), not waso/latency (legitimately < 24 min).
            if canonical_key in {"total_sleep_time", "time_in_bed"} and numeric < 24:
                numeric = numeric * 60

            # Infer unit from canonical key
            if canonical_key in _PCT_KEYS:
                unit = "%"
            elif canonical_key in _MIN_KEYS:
                unit = "min"
            elif canonical_key in {"ahi", "rdi", "arousal_index", "obstructive_ahi",
                                    "central_ahi", "rera_index", "plm_index", "odi"}:
                unit = "/hr"
            else:
                unit = ""

            flagged, flag_note = _flag(canonical_key, numeric)
            report.metrics[canonical_key] = ExtractedValue(
                numeric=numeric, unit=unit, raw=str(raw_val),
                flagged=flagged, flag_note=flag_note,
            )

            # Auto-learn: find the label that maps to this canonical key in the text
            label = _reverse_map_label(text, numeric)
            if label and label not in METRIC_ALIASES:
                _persist_learned_alias(label, canonical_key)

    except Exception:
        pass  # LLM fallback is non-critical — never block the main stream


# =============================================================================
# DASHBOARD SERIALIZATION
# =============================================================================

def has_dashboard_data(ctx: "MultiReportContext") -> bool:
    """Return True if ctx contains ≥2 numeric metric values across all reports."""
    count = 0
    all_reports = [r for pg in ctx.patient_groups for r in pg.reports] + list(ctx.unmatched_reports)
    for r in all_reports:
        count += sum(1 for v in r.metrics.values() if v.numeric is not None)
        if count >= 2:
            return True
    return False


def to_dashboard_dict(ctx: "MultiReportContext") -> dict:
    """Serialize MultiReportContext to a JSON-safe dict for the SSE metrics event."""
    patients = []
    for pg in ctx.patient_groups:
        reports = []
        for r in pg.reports:
            numeric_metrics = {
                k: {
                    "value": v.numeric,
                    "unit": v.unit,
                    "flagged": v.flagged,
                    "flag_note": v.flag_note,
                }
                for k, v in r.metrics.items()
                if v.numeric is not None
            }
            reports.append({
                "filename": r.filename,
                "date": r.study_date,
                "report_type": r.report_type,
                "metrics": numeric_metrics,
            })
        patients.append({
            "name": pg.patient_name or pg.patient_key,
            "reports": reports,
        })
    # Also include unmatched reports
    for r in ctx.unmatched_reports:
        numeric_metrics = {
            k: {
                "value": v.numeric,
                "unit": v.unit,
                "flagged": v.flagged,
                "flag_note": v.flag_note,
            }
            for k, v in r.metrics.items()
            if v.numeric is not None
        }
        patients.append({
            "name": r.patient_name or r.filename,
            "reports": [{
                "filename": r.filename,
                "date": r.study_date,
                "report_type": r.report_type,
                "metrics": numeric_metrics,
            }],
        })
    return {
        "total_reports": ctx.total_reports,
        "total_patients": ctx.total_patients,
        "patients": patients,
    }
