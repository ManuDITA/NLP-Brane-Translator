#!/usr/bin/env python3
"""
Healthcare Package for NLP-Brane-Translator

Brane entrypoint: the action name is read from sys.argv[1] (set via
container.yml command.args). Input arguments arrive as uppercase env vars
whose values are JSON-serialised by branelet (e.g. DATASET, WEIGHT_KG).

Function categories
-------------------
String-based (single patient, inline JSON via DATASET env var)
  validate_patient_data  -- basic field + range validation
  analyze_heart_disease  -- cardiovascular risk scoring
  generate_report        -- full report combining validation + CVD analysis
  compute_bmi            -- BMI from weight_kg + height_cm env vars
  assess_diabetes_risk   -- type-2 diabetes risk scoring
  triage_patient         -- urgency triage (immediate / urgent / standard)

Class-returning
  get_patient_summary    -- returns PatientSummary instance (YAML sequence)

Data-based (batch CSV; Brane mounts file and passes path via PATIENTS_FILE)
  analyze_patients_file       -- writes analysis_report.json to /result/
  filter_high_risk_patients   -- writes high_risk.csv to /result/

IntermediateResult chaining
  generate_reports_from_results -- reads analysis IR, writes per-patient
                                   JSON + HTML summary to /result/
"""
import csv
import json
import os
import sys
import yaml
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {'low': 20, 'moderate': 50, 'high': 80}

BMI_CATEGORIES = [
    (18.5, 'underweight'),
    (25.0, 'normal'),
    (30.0, 'overweight'),
    (35.0, 'obese_class_1'),
    (40.0, 'obese_class_2'),
    (float('inf'), 'obese_class_3'),
]


def _resolve_env_path(var_name: str) -> str:
    """Return the file/directory path stored in an env var.

    branelet JSON-encodes path values, so the raw value may look like
    '"/tmp/brane/abc"'.  Try JSON first, fall back to the raw string.
    """
    raw = os.environ.get(var_name, '')
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            return decoded
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _env_str(name: str) -> str:
    """Read a JSON-encoded string env var."""
    raw = os.environ.get(name.upper(), '""')
    try:
        v = json.loads(raw)
        return str(v)
    except (json.JSONDecodeError, ValueError):
        return raw


def _env_float(name: str) -> float:
    """Read a JSON-encoded numeric env var."""
    raw = os.environ.get(name.upper(), '0')
    try:
        return float(json.loads(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def _out_str(value: str) -> None:
    """Print a string output; json.dumps ensures it stays a YAML string."""
    print(f'output: {json.dumps(str(value))}', flush=True)


def _out_class(class_name: str, fields: Dict[str, Any]) -> None:
    """Print a class instance output as a YAML sequence ["ClassName", {…}].

    FullValue::Instance is serialised as a 2-element list by serde, so
    serde_yaml parses ["ClassName", {fields}] as Instance(name, map).
    """
    print(yaml.dump({'output': [class_name, fields]}), end='', flush=True)


# ---------------------------------------------------------------------------
# Core analysis logic
# ---------------------------------------------------------------------------

def _validate(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate required patient fields."""
    for field in ('patient_id', 'age', 'gender', 'vital_signs', 'lab_results'):
        if field not in data:
            return False, f'Missing required field: {field}'
    try:
        age = int(data['age'])
        if not (0 <= age <= 150):
            return False, 'Age must be between 0 and 150'
    except (ValueError, TypeError):
        return False, 'Age must be a valid integer'
    if str(data.get('gender', '')).upper() not in ('M', 'F', 'O'):
        return False, 'Gender must be M, F, or O'
    return True, ''


def _analyze_cvd(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a cardiovascular risk score and return an assessment dict."""
    factors: List[str] = []
    score = 0.0

    age = int(data.get('age', 0))
    if age >= 55:
        factors.append('Advanced age (>=55)')
        score += 20
    if str(data.get('gender', '')).upper() == 'M':
        factors.append('Male gender')
        score += 15

    vitals = data.get('vital_signs', {})
    bp = int(vitals.get('blood_pressure', 0))
    hr = int(vitals.get('heart_rate', 0))
    if bp > 140:
        factors.append('Elevated blood pressure')
        score += 25
    if hr > 100 or hr < 60:
        factors.append('Abnormal heart rate')
        score += 15

    labs = data.get('lab_results', {})
    chol = int(labs.get('total_cholesterol', 0))
    glucose = float(labs.get('glucose', 0))
    if chol > 200:
        factors.append('High total cholesterol')
        score += 20
    if glucose > 100:
        factors.append('Elevated fasting glucose')
        score += 10

    history = data.get('medical_history', [])
    if isinstance(history, str):
        history = [h for h in history.split('|') if h]
    for condition, points in (('diabetes', 15), ('hypertension', 20), ('smoking', 25)):
        if condition in history:
            factors.append(f'{condition.capitalize()} history')
            score += points

    score = min(score, 100.0)
    if score >= RISK_THRESHOLDS['high']:
        level = 'high'
    elif score >= RISK_THRESHOLDS['moderate']:
        level = 'moderate'
    else:
        level = 'low'

    recs: List[str] = []
    if level == 'high':
        recs += ['Schedule immediate specialist consultation',
                 'Undergo comprehensive cardiovascular evaluation']
    recs.append('Maintain regular health monitoring')
    recs.append('Follow a balanced diet and exercise regularly')
    if any('pressure' in f.lower() for f in factors):
        recs.append('Monitor blood pressure regularly')
    if any('cholesterol' in f.lower() for f in factors):
        recs.append('Consider cholesterol management strategies')
    if any('smoking' in f.lower() for f in factors):
        recs.append('Seek smoking cessation support')

    return {
        'patient_id': data.get('patient_id', 'unknown'),
        'risk_score': round(score, 2),
        'risk_level': level,
        'risk_factors': factors,
        'recommendations': recs,
    }


def _analyze_diabetes(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute a type-2 diabetes risk score."""
    factors: List[str] = []
    score = 0.0

    age = int(data.get('age', 0))
    if age >= 45:
        factors.append('Age >= 45')
        score += 20
    if age >= 65:
        score += 10  # extra points

    labs = data.get('lab_results', {})
    glucose = float(labs.get('glucose', 0))
    hba1c = float(labs.get('hba1c', 0))
    if glucose >= 126:
        factors.append('High fasting glucose (>=126 mg/dL)')
        score += 35
    elif glucose >= 100:
        factors.append('Impaired fasting glucose (100–125 mg/dL)')
        score += 20
    if hba1c >= 6.5:
        factors.append('HbA1c >= 6.5%')
        score += 35
    elif hba1c >= 5.7:
        factors.append('HbA1c pre-diabetic range (5.7–6.4%)')
        score += 15

    vitals = data.get('vital_signs', {})
    weight = float(vitals.get('weight_kg', 0))
    height = float(vitals.get('height_cm', 0))
    if weight > 0 and height > 0:
        bmi = weight / ((height / 100) ** 2)
        if bmi >= 30:
            factors.append(f'Obesity (BMI {bmi:.1f})')
            score += 20
        elif bmi >= 25:
            factors.append(f'Overweight (BMI {bmi:.1f})')
            score += 10

    history = data.get('medical_history', [])
    if isinstance(history, str):
        history = [h for h in history.split('|') if h]
    if 'diabetes' in history:
        factors.append('Family/personal diabetes history')
        score += 25
    if 'hypertension' in history:
        factors.append('Hypertension history')
        score += 10

    score = min(score, 100.0)
    level = 'high' if score >= 50 else ('moderate' if score >= 25 else 'low')

    return {
        'patient_id': data.get('patient_id', 'unknown'),
        'risk_score': round(score, 2),
        'risk_level': level,
        'risk_factors': factors,
    }


def _triage(data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine triage urgency from vital signs."""
    vitals = data.get('vital_signs', {})
    bp = int(vitals.get('blood_pressure', 120))
    hr = int(vitals.get('heart_rate', 70))
    temp = float(vitals.get('temperature', 37.0))
    spo2 = float(vitals.get('spo2', 98))

    flags: List[str] = []
    score = 0

    if bp >= 180 or bp < 80:
        flags.append(f'Critical blood pressure ({bp} mmHg)')
        score += 3
    elif bp >= 160:
        flags.append(f'Elevated blood pressure ({bp} mmHg)')
        score += 2

    if hr >= 130 or hr < 40:
        flags.append(f'Critical heart rate ({hr} bpm)')
        score += 3
    elif hr >= 100 or hr < 60:
        flags.append(f'Abnormal heart rate ({hr} bpm)')
        score += 1

    if temp >= 40.0 or temp < 35.0:
        flags.append(f'Critical temperature ({temp}°C)')
        score += 3
    elif temp >= 38.5:
        flags.append(f'High fever ({temp}°C)')
        score += 2

    if spo2 < 90:
        flags.append(f'Critical SpO2 ({spo2}%)')
        score += 4
    elif spo2 < 95:
        flags.append(f'Low SpO2 ({spo2}%)')
        score += 2

    if score >= 6:
        level = 'immediate'
        next_steps = ['Call emergency services', 'Administer oxygen if available',
                      'Do not leave patient unattended']
    elif score >= 3:
        level = 'urgent'
        next_steps = ['Seek emergency care within 1 hour', 'Monitor vitals continuously']
    else:
        level = 'standard'
        next_steps = ['Schedule a routine medical appointment', 'Monitor symptoms']

    return {
        'patient_id': data.get('patient_id', 'unknown'),
        'triage_level': level,
        'flags': flags,
        'next_steps': next_steps,
    }


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def action_validate_patient_data() -> None:
    raw = _env_str('DATASET')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    is_valid, error = _validate(data)
    _out_str(json.dumps({'is_valid': is_valid, 'error': error}))


def action_analyze_heart_disease() -> None:
    raw = _env_str('DATASET')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _out_str(json.dumps({'error': 'Invalid JSON', 'status': 'failed'}))
        return
    _out_str(json.dumps(_analyze_cvd(data)))


def action_generate_report() -> None:
    raw = _env_str('DATASET')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _out_str(json.dumps({'error': 'Invalid JSON', 'status': 'failed'}))
        return
    is_valid, error = _validate(data)
    if not is_valid:
        _out_str(json.dumps({'status': 'validation_failed', 'error': error}))
        return
    assessment = _analyze_cvd(data)
    report = {
        'status': 'success',
        'patient_id': data.get('patient_id', 'unknown'),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'analysis': assessment,
        'overall_risk_level': assessment['risk_level'],
    }
    _out_str(json.dumps(report))


def action_compute_bmi() -> None:
    weight = _env_float('WEIGHT_KG')
    height = _env_float('HEIGHT_CM')
    if height <= 0:
        _out_str(json.dumps({'error': 'height_cm must be > 0', 'status': 'failed'}))
        return
    bmi = weight / ((height / 100.0) ** 2)
    category = next(cat for threshold, cat in BMI_CATEGORIES if bmi < threshold)
    _out_str(json.dumps({'bmi': round(bmi, 2), 'category': category,
                         'weight_kg': weight, 'height_cm': height}))


def action_assess_diabetes_risk() -> None:
    raw = _env_str('DATASET')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _out_str(json.dumps({'error': 'Invalid JSON', 'status': 'failed'}))
        return
    _out_str(json.dumps(_analyze_diabetes(data)))


def action_triage_patient() -> None:
    raw = _env_str('DATASET')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _out_str(json.dumps({'error': 'Invalid JSON', 'status': 'failed'}))
        return
    _out_str(json.dumps(_triage(data)))


def action_get_patient_summary() -> None:
    """Return a PatientSummary class instance.

    BraneScript workflow usage:
        let p := get_patient_summary("{...}");
        println(p.risk_level);
        println(p.risk_score);

    FullValue::Instance is serialised as ["ClassName", {fields}] by serde,
    so we print a YAML 2-element list that serde_yaml deserialises correctly.
    """
    raw = _env_str('DATASET')
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _out_str(json.dumps({'error': 'Invalid JSON', 'status': 'failed'}))
        return
    assessment = _analyze_cvd(data)
    _out_class('PatientSummary', {
        'patient_id': str(data.get('patient_id', 'unknown')),
        'age': int(data.get('age', 0)),
        'gender': str(data.get('gender', 'O')),
        'risk_level': str(assessment['risk_level']),
        'risk_score': float(assessment['risk_score']),
    })


def action_analyze_patients_file() -> None:
    """Data-based: read patients CSV, write analysis_report.json to /result/."""
    path = _resolve_env_path('PATIENTS_FILE')
    if not path:
        raise RuntimeError('PATIENTS_FILE not set — invoke via a Data reference')

    patients: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history = [h for h in row.get('medical_history', '').split('|') if h]
            patients.append({
                'patient_id': row['patient_id'],
                'age': int(row['age']),
                'gender': row['gender'],
                'vital_signs': {
                    'blood_pressure': int(row.get('blood_pressure', 0)),
                    'heart_rate': int(row.get('heart_rate', 0)),
                },
                'lab_results': {
                    'total_cholesterol': int(row.get('total_cholesterol', 0)),
                    'glucose': float(row.get('glucose', 0)),
                },
                'medical_history': history,
            })

    results, errors = [], []
    for record in patients:
        is_valid, err_msg = _validate(record)
        if not is_valid:
            errors.append({'patient_id': record.get('patient_id', 'unknown'), 'error': err_msg})
        else:
            results.append(_analyze_cvd(record))

    report = {
        'total_records': len(patients),
        'processed': len(results),
        'validation_errors': len(errors),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'results': results,
        'errors': errors,
    }
    os.makedirs('/result', exist_ok=True)
    with open('/result/analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def action_filter_high_risk_patients() -> None:
    """Data-based: filter CSV to only high-risk patients, write to /result/high_risk.csv."""
    path = _resolve_env_path('PATIENTS_FILE')
    if not path:
        raise RuntimeError('PATIENTS_FILE not set — invoke via a Data reference')

    high_risk_rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            history = [h for h in row.get('medical_history', '').split('|') if h]
            record: Dict[str, Any] = {
                'patient_id': row['patient_id'],
                'age': int(row.get('age', 0)),
                'gender': row.get('gender', 'O'),
                'vital_signs': {
                    'blood_pressure': int(row.get('blood_pressure', 0)),
                    'heart_rate': int(row.get('heart_rate', 0)),
                },
                'lab_results': {
                    'total_cholesterol': int(row.get('total_cholesterol', 0)),
                    'glucose': float(row.get('glucose', 0)),
                },
                'medical_history': history,
            }
            is_valid, _ = _validate(record)
            if is_valid:
                assessment = _analyze_cvd(record)
                if assessment['risk_level'] == 'high':
                    high_risk_rows.append(dict(row, risk_score=str(assessment['risk_score'])))

    os.makedirs('/result', exist_ok=True)
    out_fields = fieldnames + (['risk_score'] if 'risk_score' not in fieldnames else [])
    with open('/result/high_risk.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(high_risk_rows)

    summary = {
        'high_risk_count': len(high_risk_rows),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    with open('/result/filter_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def action_generate_reports_from_results() -> None:
    """IntermediateResult chaining: read analysis dir, write per-patient reports + HTML."""
    result_dir = _resolve_env_path('ANALYSIS')
    if not result_dir:
        raise RuntimeError('ANALYSIS not set — invoke via an IntermediateResult reference')

    # Find analysis_report.json anywhere in the result dir tree
    analysis_path = None
    for root, _, files in os.walk(result_dir):
        if 'analysis_report.json' in files:
            analysis_path = os.path.join(root, 'analysis_report.json')
            break
    if analysis_path is None:
        raise FileNotFoundError(f'analysis_report.json not found under {result_dir}')

    with open(analysis_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    reports_dir = '/result/reports'
    os.makedirs(reports_dir, exist_ok=True)

    for entry in data.get('results', []):
        pid = entry.get('patient_id', 'unknown')
        report = {
            'patient_id': pid,
            'generated_at': ts,
            'risk_level': entry.get('risk_level'),
            'risk_score': entry.get('risk_score'),
            'risk_factors': entry.get('risk_factors', []),
            'recommendations': entry.get('recommendations', []),
        }
        with open(f'{reports_dir}/{pid}_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

    rows = ''.join(
        f"<tr><td>{e.get('patient_id')}</td><td>{e.get('risk_level')}</td>"
        f"<td>{e.get('risk_score')}</td></tr>"
        for e in data.get('results', [])
    )
    html = (
        '<!DOCTYPE html><html><head><title>Healthcare Analysis</title></head><body>'
        '<h1>Healthcare Analysis Report</h1>'
        f'<p>Generated: {ts}</p>'
        f'<p>Total records: {data.get("total_records", 0)} | '
        f'Processed: {data.get("processed", 0)} | '
        f'Errors: {data.get("validation_errors", 0)}</p>'
        '<table border="1"><tr><th>Patient ID</th><th>Risk Level</th><th>Score</th></tr>'
        f'{rows}</table></body></html>'
    )
    with open('/result/summary.html', 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ACTIONS = {
    'validate_patient_data':        action_validate_patient_data,
    'analyze_heart_disease':        action_analyze_heart_disease,
    'generate_report':              action_generate_report,
    'compute_bmi':                  action_compute_bmi,
    'assess_diabetes_risk':         action_assess_diabetes_risk,
    'triage_patient':               action_triage_patient,
    'get_patient_summary':          action_get_patient_summary,
    'analyze_patients_file':        action_analyze_patients_file,
    'filter_high_risk_patients':    action_filter_high_risk_patients,
    'generate_reports_from_results': action_generate_reports_from_results,
}


def main() -> None:
    if len(sys.argv) < 2:
        _out_str(json.dumps({'error': 'No action name in argv[1]', 'status': 'failed'}))
        sys.exit(1)
    action = sys.argv[1]
    handler = _ACTIONS.get(action)
    if handler is None:
        _out_str(json.dumps({'error': f'Unknown action: {action!r}', 'status': 'failed'}))
        sys.exit(1)
    handler()


if __name__ == '__main__':
    main()