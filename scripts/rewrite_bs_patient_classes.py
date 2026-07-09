#!/usr/bin/env python3
"""Rewrite BraneScript patient examples to Patient class construction syntax."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

JSONL_FILES = [
    REPO / "data/examples/basic.jsonl",
    REPO / "data/examples/classes_arrays.jsonl",
    REPO / "data/examples/control_flow.jsonl",
    REPO / "data/examples/functions.jsonl",
    REPO / "data/examples/healthcare.jsonl",
    REPO / "data/examples/advanced.jsonl",
    REPO / "data/examples/packages.jsonl",
    REPO / "data/examples/training_500.jsonl",
    REPO / "data/training/execution_results.jsonl",
]

LOCAL_CLASS_RE = re.compile(
    r"(?:^|\n)\s*class (?:VitalSigns|LabResults|Patient|RiskAssessment|TriageResult)\s*\{.*?\}\s*(?=\n|$)",
    re.DOTALL,
)
PATIENT_STRING_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*"(\{(?:[^"\\]|\\.)*\})"\s*;',
    re.DOTALL,
)
SIMPLE_STRING_RE = re.compile(r'let\s+(\w+)\s*:=\s*"((?:[^"\\]|\\.)*)"\s*;')
VITALS_BLOCK_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*new\s+VitalSigns\s*\{\s*(.*?)\s*\}\s*;',
    re.DOTALL,
)
LABS_BLOCK_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*new\s+LabResults\s*\{\s*(.*?)\s*\}\s*;',
    re.DOTALL,
)
PATIENT_BLOCK_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*new\s+Patient\s*\{\s*(.*?)\s*\}\s*;',
    re.DOTALL,
)
FIELD_LINE_RE = re.compile(r'^\s*(\w+)\s*:=\s*(.+?)\s*,\s*$')
ASSIGN_CALL_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*'
    r'(analyze_heart_disease|assess_diabetes_risk|triage_patient|get_patient_summary)\s*\(',
)
PARALLEL_ASSIGN_RE = re.compile(r'let\s+(\w+)\s*:=\s*parallel\b.*?\];', re.DOTALL)
TARGET_PATIENT_FNS = (
    'validate_patient_data',
    'analyze_heart_disease',
    'generate_report',
    'assess_diabetes_risk',
    'triage_patient',
    'get_patient_summary',
)


def strip_local_class_defs(bs: str) -> str:
    """Remove local class definitions now provided by the healthcare package."""
    return LOCAL_CLASS_RE.sub('\n', bs)


def bs_string(value: object) -> str:
    return json.dumps('' if value is None else str(value), ensure_ascii=False)


def parse_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_int(value: object, default: int = 0) -> str:
    return str(parse_int(value, default))


def format_real(value: object, default: float = 0.0) -> str:
    number = parse_float(value, default)
    if number.is_integer():
        return f'{int(number)}.0'
    return str(number)


def decode_bs_string(value: str) -> str | None:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        try:
            return value.replace('\\"', '"').replace('\\\\', '\\')
        except Exception:
            return None


def try_parse_patient_json(json_str: str) -> dict | None:
    """Attempt to parse the JSON from inside the escaped BraneScript string."""
    try:
        unescaped = json.loads(f'"{json_str}"')
        parsed = json.loads(unescaped)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        try:
            unescaped = json_str.replace('\\"', '"').replace('\\\\', '\\')
            parsed = json.loads(unescaped)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None


def build_patient_dict(patient_json: dict, patient_id: str | None = None) -> dict:
    """Normalize raw patient JSON/class fields into a single patient dict."""
    vs = patient_json.get('vital_signs', {}) if isinstance(patient_json.get('vital_signs', {}), dict) else {}
    lr = patient_json.get('lab_results', {}) if isinstance(patient_json.get('lab_results', {}), dict) else {}
    medical_history = patient_json.get('medical_history', [])
    if isinstance(medical_history, list):
        history = ','.join(str(item) for item in medical_history)
    elif medical_history in (None, ''):
        history = ''
    else:
        history = str(medical_history)

    return {
        'patient_id': patient_id or patient_json.get('patient_id', 'unknown'),
        'age': patient_json.get('age', 45),
        'gender': patient_json.get('gender', 'M'),
        'weight': vs.get('weight_kg', patient_json.get('weight_kg', patient_json.get('weight', 70))),
        'height': vs.get('height_cm', patient_json.get('height_cm', patient_json.get('height', 170))),
        'medical_history': history,
        'vital_signs': {
            'blood_pressure': vs.get('blood_pressure', patient_json.get('blood_pressure', 120)),
            'heart_rate': vs.get('heart_rate', patient_json.get('heart_rate', 70)),
            'spo2': vs.get('spo2', patient_json.get('spo2', 98)),
            'temperature': vs.get('temperature', patient_json.get('temperature', 37.0)),
        },
        'lab_results': {
            'cholesterol': lr.get('total_cholesterol', lr.get('cholesterol', patient_json.get('cholesterol', 0))),
            'glucose': lr.get('glucose', patient_json.get('glucose', 0)),
            'hba1c': lr.get('hba1c', patient_json.get('hba1c', 0.0)),
        },
    }


def build_patient_block(patient_json: dict, var_name: str = 'patient') -> str:
    """Build BraneScript construction syntax for VitalSigns/LabResults/Patient."""
    patient = build_patient_dict(patient_json)
    suffix = f'_{var_name}' if var_name != 'patient' else ''
    vitals_var = f'vitals{suffix}'
    labs_var = f'labs{suffix}'
    vitals = patient['vital_signs']
    labs = patient['lab_results']

    lines = [
        f'let {vitals_var} := new VitalSigns {{',
        f'    blood_pressure := {format_int(vitals.get("blood_pressure"), 120)},',
        f'    heart_rate     := {format_int(vitals.get("heart_rate"), 70)},',
        f'    spo2           := {format_int(vitals.get("spo2"), 98)},',
        f'    temperature    := {format_real(vitals.get("temperature"), 37.0)},',
        f'}};',
        f'let {labs_var} := new LabResults {{',
        f'    cholesterol := {format_int(labs.get("cholesterol"), 0)},',
        f'    glucose     := {format_int(labs.get("glucose"), 0)},',
        f'    hba1c       := {format_real(labs.get("hba1c"), 0.0)},',
        f'}};',
        f'let {var_name} := new Patient {{',
        f'    patient_id      := {bs_string(patient.get("patient_id", "unknown"))},',
        f'    age             := {format_int(patient.get("age"), 45)},',
        f'    gender          := {bs_string(patient.get("gender", "M"))},',
        f'    weight          := {format_int(patient.get("weight"), 70)},',
        f'    height          := {format_int(patient.get("height"), 170)},',
        f'    medical_history := {bs_string(patient.get("medical_history", ""))},',
        f'    vital_signs     := {vitals_var},',
        f'    lab_results     := {labs_var},',
        f'}};',
    ]
    return '\n'.join(lines)


def build_default_patient_block(patient_id: str, var_name: str = 'patient') -> str:
    return build_patient_block({'patient_id': patient_id}, var_name=var_name)


def parse_block_fields(block_body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block_body.splitlines():
        match = FIELD_LINE_RE.match(line)
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def expr_or_default(raw: str | None, default_expr: str) -> str:
    return raw.strip() if raw else default_expr


def expr_int(raw: str | None, default: int) -> str:
    if raw is None:
        return str(default)
    raw = raw.strip()
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


def expr_real(raw: str | None, default: float) -> str:
    if raw is None:
        return format_real(default, default)
    raw = raw.strip()
    try:
        return format_real(float(raw), default)
    except ValueError:
        return raw


def normalize_vitals_blocks(bs: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        var_name = match.group(1)
        fields = parse_block_fields(match.group(2))
        lines = [
            f'let {var_name} := new VitalSigns {{',
            f'    blood_pressure := {expr_int(fields.get("blood_pressure"), 120)},',
            f'    heart_rate     := {expr_int(fields.get("heart_rate"), 70)},',
            f'    spo2           := {expr_int(fields.get("spo2"), 98)},',
            f'    temperature    := {expr_real(fields.get("temperature"), 37.0)},',
            f'}};',
        ]
        return '\n'.join(lines)

    return VITALS_BLOCK_RE.sub(repl, bs), count


def normalize_labs_blocks(bs: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        var_name = match.group(1)
        fields = parse_block_fields(match.group(2))
        cholesterol = fields.get('cholesterol', fields.get('total_cholesterol'))
        lines = [
            f'let {var_name} := new LabResults {{',
            f'    cholesterol := {expr_int(cholesterol, 0)},',
            f'    glucose     := {expr_int(fields.get("glucose"), 0)},',
            f'    hba1c       := {expr_real(fields.get("hba1c"), 0.0)},',
            f'}};',
        ]
        return '\n'.join(lines)

    return LABS_BLOCK_RE.sub(repl, bs), count


def normalize_patient_blocks(bs: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        var_name = match.group(1)
        fields = parse_block_fields(match.group(2))
        lines = [
            f'let {var_name} := new Patient {{',
            f'    patient_id      := {expr_or_default(fields.get("patient_id"), bs_string("unknown"))},',
            f'    age             := {expr_int(fields.get("age"), 45)},',
            f'    gender          := {expr_or_default(fields.get("gender"), bs_string("M"))},',
            f'    weight          := {expr_int(fields.get("weight"), 70)},',
            f'    height          := {expr_int(fields.get("height"), 170)},',
            f'    medical_history := {expr_or_default(fields.get("medical_history"), bs_string(""))},',
            f'    vital_signs     := {expr_or_default(fields.get("vital_signs"), "vitals")},',
            f'    lab_results     := {expr_or_default(fields.get("lab_results"), "labs")},',
            f'}};',
        ]
        return '\n'.join(lines)

    return PATIENT_BLOCK_RE.sub(repl, bs), count


def rewrite_patient_string_assignments(bs: str) -> tuple[str, int]:
    rewrites = 0

    def replace_match(match: re.Match[str]) -> str:
        nonlocal rewrites
        var_name = match.group(1)
        patient = try_parse_patient_json(match.group(2))
        if patient is None or ('patient_id' not in patient and 'vital_signs' not in patient):
            return match.group(0)
        rewrites += 1
        return build_patient_block(patient, var_name)

    return PATIENT_STRING_RE.sub(replace_match, bs), rewrites


def rewrite_simple_patient_id_assignments(bs: str) -> tuple[str, int]:
    rewrites = 0

    def replace_match(match: re.Match[str]) -> str:
        nonlocal rewrites
        var_name = match.group(1)
        decoded = decode_bs_string(match.group(2))
        if decoded is None or decoded.startswith('{'):
            return match.group(0)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', decoded):
            return match.group(0)
        if not re.search(rf'\b(?:{"|".join(TARGET_PATIENT_FNS)})\s*\(\s*{re.escape(var_name)}\s*\)', bs):
            return match.group(0)
        rewrites += 1
        return build_default_patient_block(decoded, var_name)

    return SIMPLE_STRING_RE.sub(replace_match, bs), rewrites


def replace_raw_prints(bs: str, variable_names: set[str], fields: tuple[str, str]) -> str:
    for var_name in variable_names:
        bs = re.sub(
            rf'println\s*\(\s*{re.escape(var_name)}\s*\)\s*;',
            f'println({var_name}.{fields[0]});\nprintln({var_name}.{fields[1]});',
            bs,
        )
    return bs


def replace_parallel_raw_prints(bs: str, array_name: str, fields: tuple[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        expr = match.group(1)
        return f'println({expr}.{fields[0]});\nprintln({expr}.{fields[1]});'

    return re.sub(
        rf'println\s*\(\s*({re.escape(array_name)}\s*\[\s*\d+\s*\])\s*\)\s*;',
        repl,
        bs,
    )


def fix_output_fields(bs: str) -> str:
    """Update raw println(result) calls for class-returning healthcare functions."""
    risk_vars = {
        var_name
        for var_name, fn_name in ASSIGN_CALL_RE.findall(bs)
        if fn_name in {'analyze_heart_disease', 'assess_diabetes_risk'}
    }
    triage_vars = {var_name for var_name, fn_name in ASSIGN_CALL_RE.findall(bs) if fn_name == 'triage_patient'}
    summary_vars = {var_name for var_name, fn_name in ASSIGN_CALL_RE.findall(bs) if fn_name == 'get_patient_summary'}

    bs = replace_raw_prints(bs, risk_vars, ('risk_level', 'risk_score'))
    bs = replace_raw_prints(bs, triage_vars, ('triage_level', 'recommendation'))
    bs = replace_raw_prints(bs, summary_vars, ('risk_level', 'risk_score'))

    for match in PARALLEL_ASSIGN_RE.finditer(bs):
        array_name = match.group(1)
        block = match.group(0)
        if 'triage_patient(' in block:
            bs = replace_parallel_raw_prints(bs, array_name, ('triage_level', 'recommendation'))
        elif 'analyze_heart_disease(' in block or 'assess_diabetes_risk(' in block:
            bs = replace_parallel_raw_prints(bs, array_name, ('risk_level', 'risk_score'))

    return bs


def rewrite_branescript(bs: str) -> tuple[str, int]:
    """Rewrite one BraneScript snippet. Returns (new_bs, rewrite_count)."""
    if 'import healthcare' not in bs:
        return bs, 0

    rewrites = 0
    original = bs

    bs = strip_local_class_defs(bs)

    bs, count = rewrite_patient_string_assignments(bs)
    rewrites += count

    bs, count = rewrite_simple_patient_id_assignments(bs)
    rewrites += count

    bs, count = normalize_vitals_blocks(bs)
    rewrites += count

    bs, count = normalize_labs_blocks(bs)
    rewrites += count

    bs, count = normalize_patient_blocks(bs)
    rewrites += count

    bs = fix_output_fields(bs)
    bs = re.sub(r'\n{3,}', '\n\n', bs).strip() + '\n'

    if bs == original:
        return original, 0
    return bs, max(rewrites, 1)


def process_file(path: Path) -> tuple[int, int]:
    """Process one JSONL file. Returns (entries_processed, rewrites_made)."""
    if not path.exists():
        print(f'  SKIP (not found): {path.name}')
        return 0, 0

    lines = path.read_text(encoding='utf-8').splitlines()
    out_lines: list[str] = []
    total_entries = 0
    total_rewrites = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue

        bs = entry.get('branescript', '')
        if bs:
            new_bs, rewrite_count = rewrite_branescript(bs)
            if new_bs != bs:
                entry['branescript'] = new_bs
                total_rewrites += rewrite_count
        out_lines.append(json.dumps(entry, ensure_ascii=False))
        total_entries += 1

    path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    return total_entries, total_rewrites


def main() -> None:
    total_entries = 0
    total_rewrites = 0
    for path in JSONL_FILES:
        entries, rewrites = process_file(path)
        print(f'  {path.name}: {entries} entries, {rewrites} rewrites')
        total_entries += entries
        total_rewrites += rewrites
    print(f'\nTotal: {total_entries} entries processed, {total_rewrites} BraneScripts rewritten')


if __name__ == '__main__':
    main()
