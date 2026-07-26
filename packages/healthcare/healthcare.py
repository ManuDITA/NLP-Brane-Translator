#!/usr/bin/env python3
"""
Healthcare Package for NLP-Brane-Translator

Brane entrypoint: the action name is read from sys.argv[1] (set via
container.yml command.args). Input arguments arrive as uppercase env vars
whose values are JSON-serialised by branelet (e.g. PATIENT, WEIGHT_KG).

Function categories
-------------------
Single-patient class-based (Patient via PATIENT env var)
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
  batch_diabetes_from_file    -- writes diabetes_report.json to /result/
  batch_triage_from_file      -- writes triage_report.json to /result/
  compute_cohort_statistics   -- writes cohort_stats.json to /result/
  filter_by_condition         -- writes condition_patients.csv + summary to /result/

IntermediateResult chaining
  generate_reports_from_results -- reads analysis IR, writes per-patient
                                   JSON + HTML summary to /result/
  compute_risk_distribution     -- reads analysis IR, returns JSON string

Additional string-based utilities
  compute_mortality_risk      -- composite mortality risk score
  check_vital_signs           -- per-vital clinical interpretation
  predict_readmission_risk    -- 30-day readmission risk score
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


def _parse_patient_class() -> Dict[str, Any]:
    """Parse a Patient class instance from the PATIENT env var.

    Brane serializes class inputs as JSON: ["ClassName", {fields}]
    Nested classes (VitalSigns, LabResults) use the same format.
    Maps BraneScript field names to the internal dict format expected
    by the analysis helpers.
    """
    raw = os.environ.get("PATIENT", "")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}

    if isinstance(parsed, list) and len(parsed) == 2:
        _, fields = parsed
    else:
        fields = parsed if isinstance(parsed, dict) else {}

    # Unwrap nested VitalSigns
    vs = fields.get("vital_signs", {})
    if isinstance(vs, list) and len(vs) == 2:
        vs = vs[1]
    if not isinstance(vs, dict):
        vs = {}

    # Unwrap nested LabResults
    lr = fields.get("lab_results", {})
    if isinstance(lr, list) and len(lr) == 2:
        lr = lr[1]
    if not isinstance(lr, dict):
        lr = {}

    # medical_history: stored as comma-separated string in BS, convert back to list
    mh_raw = fields.get("medical_history", "")
    if isinstance(mh_raw, str):
        medical_history = [h.strip() for h in mh_raw.split(",") if h.strip()]
    elif isinstance(mh_raw, list):
        medical_history = mh_raw
    else:
        medical_history = []

    return {
        "patient_id": str(fields.get("patient_id", "unknown")),
        "age": int(fields.get("age", 0)),
        "gender": str(fields.get("gender", "")),
        "vital_signs": {
            "blood_pressure": int(vs.get("blood_pressure", 120)),
            "heart_rate": int(vs.get("heart_rate", 70)),
            "temperature": float(vs.get("temperature", 37.0)),
            "spo2": float(vs.get("spo2", 98)),
            # weight_kg and height_cm are on the Patient, not VitalSigns
            "weight_kg": int(fields.get("weight", 0)),
            "height_cm": int(fields.get("height", 0)),
        },
        "lab_results": {
            "total_cholesterol": int(lr.get("cholesterol", 0)),
            "glucose": int(lr.get("glucose", 0)),
            "hba1c": float(lr.get("hba1c", 0)),
        },
        "medical_history": medical_history,
    }


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
    data = _parse_patient_class()
    is_valid, error = _validate(data)
    _out_str(json.dumps({'is_valid': is_valid, 'error': error}))


def action_analyze_heart_disease() -> None:
    data = _parse_patient_class()
    result = _analyze_cvd(data)
    factors = result.get('risk_factors', [])
    _out_class('RiskAssessment', {
        'patient_id': result.get('patient_id', 'unknown'),
        'risk_score': result.get('risk_score', 0.0),
        'risk_level': result.get('risk_level', 'unknown'),
        'top_factor': factors[0] if factors else 'none',
    })


def action_generate_report() -> None:
    data = _parse_patient_class()
    is_valid, error = _validate(data)
    if not is_valid:
        _out_str(json.dumps({'status': 'validation_failed', 'error': error}))
        return
    assessment = _analyze_cvd(data)
    report = {
        'status': 'success',
        'patient_id': data.get('patient_id', 'unknown'),
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
    data = _parse_patient_class()
    result = _analyze_diabetes(data)
    factors = result.get('risk_factors', [])
    _out_class('RiskAssessment', {
        'patient_id': result.get('patient_id', 'unknown'),
        'risk_score': result.get('risk_score', 0.0),
        'risk_level': result.get('risk_level', 'unknown'),
        'top_factor': factors[0] if factors else 'none',
    })


def action_triage_patient() -> None:
    data = _parse_patient_class()
    result = _triage(data)
    steps = result.get('next_steps', [])
    _out_class('TriageResult', {
        'patient_id': result.get('patient_id', 'unknown'),
        'triage_level': result.get('triage_level', 'standard'),
        'recommendation': steps[0] if steps else 'Monitor symptoms',
    })


def action_get_patient_summary() -> None:
    """Return a PatientSummary class instance.

    BraneScript workflow usage:
        let p := get_patient_summary(patient);
        println(p.risk_level);
        println(p.risk_score);

    FullValue::Instance is serialised as ["ClassName", {fields}] by serde,
    so we print a YAML 2-element list that serde_yaml deserialises correctly.
    """
    data = _parse_patient_class()
    assessment = _analyze_cvd(data)
    _out_class('PatientSummary', {
        'patient_id': str(data.get('patient_id', 'unknown')),
        'age': int(data.get('age', 0)),
        'gender': str(data.get('gender', 'O')),
        'risk_level': str(assessment['risk_level']),
        'risk_score': float(assessment['risk_score']),
    })


def _read_patients_csv(path: str) -> List[Dict[str, Any]]:
    """Read a patients CSV and return a list of structured patient dicts."""
    patients: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history = [h for h in row.get('medical_history', '').split('|') if h]
            patients.append({
                'patient_id': row['patient_id'],
                'age': int(row.get('age', 0) or 0),
                'gender': row.get('gender', 'O'),
                'vital_signs': {
                    'blood_pressure': int(row.get('blood_pressure', 0) or 0),
                    'heart_rate': int(row.get('heart_rate', 0) or 0),
                    'temperature': float(row.get('temperature', 37.0) or 37.0),
                    'spo2': float(row.get('spo2', 98.0) or 98.0),
                    'weight_kg': float(row.get('weight_kg', 0) or 0),
                    'height_cm': float(row.get('height_cm', 0) or 0),
                },
                'lab_results': {
                    'total_cholesterol': int(row.get('total_cholesterol', 0) or 0),
                    'glucose': float(row.get('glucose', 0) or 0),
                    'hba1c': float(row.get('hba1c', 0) or 0),
                },
                'medical_history': history,
            })
    return patients


def action_analyze_patients_file() -> None:
    """Data-based: read patients CSV, write analysis_report.json to /result/."""
    path = _resolve_env_path('PATIENTS_FILE')
    if not path:
        raise RuntimeError('PATIENTS_FILE not set — invoke via a Data reference')
    patients = _read_patients_csv(path)
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

    patients = _read_patients_csv(path)
    fieldnames = [
        'patient_id', 'age', 'gender', 'blood_pressure', 'heart_rate',
        'total_cholesterol', 'glucose', 'hba1c', 'temperature', 'spo2',
        'weight_kg', 'height_cm', 'medical_history',
    ]
    high_risk_rows: List[Dict[str, str]] = []
    for record in patients:
        is_valid, _ = _validate(record)
        if is_valid:
            assessment = _analyze_cvd(record)
            if assessment['risk_level'] == 'high':
                vitals = record.get('vital_signs', {})
                labs = record.get('lab_results', {})
                high_risk_rows.append({
                    'patient_id': str(record.get('patient_id', 'unknown')),
                    'age': str(record.get('age', 0)),
                    'gender': str(record.get('gender', 'O')),
                    'blood_pressure': str(vitals.get('blood_pressure', 0)),
                    'heart_rate': str(vitals.get('heart_rate', 0)),
                    'total_cholesterol': str(labs.get('total_cholesterol', 0)),
                    'glucose': str(labs.get('glucose', 0)),
                    'hba1c': str(labs.get('hba1c', 0)),
                    'temperature': str(vitals.get('temperature', 0)),
                    'spo2': str(vitals.get('spo2', 0)),
                    'weight_kg': str(vitals.get('weight_kg', 0)),
                    'height_cm': str(vitals.get('height_cm', 0)),
                    'medical_history': '|'.join(record.get('medical_history', [])),
                    'risk_score': str(assessment['risk_score']),
                })

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


def action_batch_diabetes_from_file() -> None:
    """Data-based: run diabetes risk assessment on all patients in CSV, write to /result/diabetes_report.json."""
    path = _resolve_env_path('PATIENTS_FILE')
    if not path:
        raise RuntimeError('PATIENTS_FILE not set')
    patients = _read_patients_csv(path)
    results, errors = [], []
    for record in patients:
        is_valid, err_msg = _validate(record)
        if not is_valid:
            errors.append({'patient_id': record.get('patient_id', 'unknown'), 'error': err_msg})
        else:
            results.append(_analyze_diabetes(record))
    report = {
        'total_records': len(patients),
        'processed': len(results),
        'validation_errors': len(errors),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'results': results,
        'errors': errors,
    }
    os.makedirs('/result', exist_ok=True)
    with open('/result/diabetes_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def action_batch_triage_from_file() -> None:
    """Data-based: triage all patients in CSV, sort by urgency, write to /result/triage_report.json."""
    path = _resolve_env_path('PATIENTS_FILE')
    if not path:
        raise RuntimeError('PATIENTS_FILE not set')
    patients = _read_patients_csv(path)
    results = []
    for record in patients:
        triage = _triage(record)
        results.append(triage)

    order = {'immediate': 0, 'urgent': 1, 'standard': 2}
    results.sort(key=lambda r: order.get(r.get('triage_level', 'standard'), 2))

    counts = {'immediate': 0, 'urgent': 0, 'standard': 0}
    for r in results:
        level = r.get('triage_level', 'standard')
        counts[level] = counts.get(level, 0) + 1

    report = {
        'total_records': len(patients),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'summary': counts,
        'patients': results,
    }
    os.makedirs('/result', exist_ok=True)
    with open('/result/triage_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)


def _statistics_for_column(values: List[float]) -> Dict[str, float]:
    """Compute basic descriptive statistics for a list of numeric values."""
    if not values:
        return {'count': 0, 'mean': 0, 'std_dev': 0, 'min': 0, 'max': 0, 'median': 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    std_dev = variance ** 0.5
    sorted_vals = sorted(values)
    mid = n // 2
    median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2 if n % 2 == 0 else sorted_vals[mid]
    return {
        'count': n,
        'mean': round(mean, 2),
        'std_dev': round(std_dev, 2),
        'min': round(min(values), 2),
        'max': round(max(values), 2),
        'median': round(median, 2),
    }


def action_compute_cohort_statistics() -> None:
    """Data-based: compute descriptive statistics for all numeric fields, write to /result/cohort_stats.json."""
    path = _resolve_env_path('PATIENTS_FILE')
    if not path:
        raise RuntimeError('PATIENTS_FILE not set')
    patients = _read_patients_csv(path)

    ages, bps, hrs, chols, glucoses, hba1cs, weights, heights, spox, temps = \
        [], [], [], [], [], [], [], [], [], []

    for p in patients:
        ages.append(float(p['age']))
        vs = p.get('vital_signs', {})
        bps.append(float(vs.get('blood_pressure', 0)))
        hrs.append(float(vs.get('heart_rate', 0)))
        temps.append(float(vs.get('temperature', 0)))
        spox.append(float(vs.get('spo2', 0)))
        weights.append(float(vs.get('weight_kg', 0)))
        heights.append(float(vs.get('height_cm', 0)))
        lr = p.get('lab_results', {})
        chols.append(float(lr.get('total_cholesterol', 0)))
        glucoses.append(float(lr.get('glucose', 0)))
        hba1cs.append(float(lr.get('hba1c', 0)))

    stats = {
        'total_patients': len(patients),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'age': _statistics_for_column(ages),
        'blood_pressure': _statistics_for_column([v for v in bps if v > 0]),
        'heart_rate': _statistics_for_column([v for v in hrs if v > 0]),
        'temperature': _statistics_for_column([v for v in temps if v > 0]),
        'spo2': _statistics_for_column([v for v in spox if v > 0]),
        'weight_kg': _statistics_for_column([v for v in weights if v > 0]),
        'height_cm': _statistics_for_column([v for v in heights if v > 0]),
        'total_cholesterol': _statistics_for_column([v for v in chols if v > 0]),
        'glucose': _statistics_for_column([v for v in glucoses if v > 0]),
        'hba1c': _statistics_for_column([v for v in hba1cs if v > 0]),
    }
    os.makedirs('/result', exist_ok=True)
    with open('/result/cohort_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)


def action_filter_by_condition() -> None:
    """Data-based: filter patients CSV to those with a given condition in medical_history.

    Input: patients_file (Data), condition (string).
    Writes: /result/condition_patients.csv and /result/filter_summary.json
    """
    path = _resolve_env_path('PATIENTS_FILE')
    condition = _env_str('CONDITION').lower().strip()
    if not path:
        raise RuntimeError('PATIENTS_FILE not set')

    matched_rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            history = [h.lower() for h in row.get('medical_history', '').split('|') if h]
            if condition in history:
                matched_rows.append(dict(row))

    os.makedirs('/result', exist_ok=True)
    with open('/result/condition_patients.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(matched_rows)

    summary = {
        'condition': condition,
        'matched_count': len(matched_rows),
        'patient_ids': [r.get('patient_id', 'unknown') for r in matched_rows],
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    with open('/result/filter_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def action_compute_risk_distribution() -> None:
    """IntermediateResult chaining: read analysis/diabetes IR, return JSON risk level distribution string."""
    result_dir = _resolve_env_path('ANALYSIS')
    if not result_dir:
        raise RuntimeError('ANALYSIS not set')

    report_path = None
    for fname in ('analysis_report.json', 'diabetes_report.json'):
        for root, _, files in os.walk(result_dir):
            if fname in files:
                report_path = os.path.join(root, fname)
                break
        if report_path:
            break

    if report_path is None:
        raise FileNotFoundError(f'No analysis_report.json or diabetes_report.json found under {result_dir}')

    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    distribution: Dict[str, int] = {'low': 0, 'moderate': 0, 'high': 0}
    for entry in data.get('results', []):
        level = entry.get('risk_level', 'unknown')
        distribution[level] = distribution.get(level, 0) + 1

    total = sum(distribution.values())
    percentages = {
        k: round(v / total * 100, 1) if total > 0 else 0.0
        for k, v in distribution.items()
    }

    _out_str(json.dumps({
        'total_patients': total,
        'distribution': distribution,
        'percentages': percentages,
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }))


def action_compute_mortality_risk() -> None:
    """Single patient: composite mortality risk combining CVD, diabetes, and triage scores."""
    data = _parse_patient_class()

    cvd = _analyze_cvd(data)
    dia = _analyze_diabetes(data)
    tri = _triage(data)

    triage_weight = {'immediate': 40, 'urgent': 20, 'standard': 0}
    triage_score = triage_weight.get(tri.get('triage_level', 'standard'), 0)

    composite = (cvd['risk_score'] * 0.4) + (dia['risk_score'] * 0.3) + (triage_score * 0.3)
    composite = round(min(composite, 100.0), 2)

    if composite >= 70:
        level = 'critical'
    elif composite >= 45:
        level = 'high'
    elif composite >= 25:
        level = 'moderate'
    else:
        level = 'low'

    _out_str(json.dumps({
        'patient_id': data.get('patient_id', 'unknown'),
        'mortality_risk_score': composite,
        'mortality_risk_level': level,
        'cvd_score': cvd['risk_score'],
        'diabetes_score': dia['risk_score'],
        'triage_level': tri.get('triage_level', 'standard'),
        'top_cvd_factor': cvd['risk_factors'][0] if cvd['risk_factors'] else 'none',
        'top_diabetes_factor': dia['risk_factors'][0] if dia['risk_factors'] else 'none',
    }))


def action_check_vital_signs() -> None:
    """Single patient: flag all abnormal vital signs with clinical interpretation."""
    data = _parse_patient_class()

    vitals = data.get('vital_signs', {})
    findings: List[Dict[str, str]] = []

    bp = int(vitals.get('blood_pressure', 0))
    if bp > 0:
        if bp >= 180:
            findings.append({'sign': 'blood_pressure', 'value': str(bp), 'status': 'hypertensive_crisis', 'note': 'Immediate medical attention required'})
        elif bp >= 140:
            findings.append({'sign': 'blood_pressure', 'value': str(bp), 'status': 'high', 'note': 'Stage 2 hypertension'})
        elif bp >= 130:
            findings.append({'sign': 'blood_pressure', 'value': str(bp), 'status': 'elevated', 'note': 'Stage 1 hypertension'})
        elif bp < 90:
            findings.append({'sign': 'blood_pressure', 'value': str(bp), 'status': 'low', 'note': 'Hypotension — monitor closely'})
        else:
            findings.append({'sign': 'blood_pressure', 'value': str(bp), 'status': 'normal', 'note': 'Within normal range'})

    hr = int(vitals.get('heart_rate', 0))
    if hr > 0:
        if hr >= 130:
            findings.append({'sign': 'heart_rate', 'value': str(hr), 'status': 'critical_high', 'note': 'Severe tachycardia'})
        elif hr >= 100:
            findings.append({'sign': 'heart_rate', 'value': str(hr), 'status': 'high', 'note': 'Tachycardia'})
        elif hr < 40:
            findings.append({'sign': 'heart_rate', 'value': str(hr), 'status': 'critical_low', 'note': 'Severe bradycardia'})
        elif hr < 60:
            findings.append({'sign': 'heart_rate', 'value': str(hr), 'status': 'low', 'note': 'Bradycardia'})
        else:
            findings.append({'sign': 'heart_rate', 'value': str(hr), 'status': 'normal', 'note': 'Within normal range'})

    temp = float(vitals.get('temperature', 0))
    if temp > 0:
        if temp >= 40.0:
            findings.append({'sign': 'temperature', 'value': str(temp), 'status': 'critical_high', 'note': 'Hyperpyrexia — critical'})
        elif temp >= 38.5:
            findings.append({'sign': 'temperature', 'value': str(temp), 'status': 'high', 'note': 'High fever'})
        elif temp >= 37.5:
            findings.append({'sign': 'temperature', 'value': str(temp), 'status': 'elevated', 'note': 'Low-grade fever'})
        elif temp < 35.0:
            findings.append({'sign': 'temperature', 'value': str(temp), 'status': 'low', 'note': 'Hypothermia'})
        else:
            findings.append({'sign': 'temperature', 'value': str(temp), 'status': 'normal', 'note': 'Within normal range'})

    spo2 = float(vitals.get('spo2', 0))
    if spo2 > 0:
        if spo2 < 90:
            findings.append({'sign': 'spo2', 'value': str(spo2), 'status': 'critical_low', 'note': 'Severe hypoxia — oxygen therapy required'})
        elif spo2 < 95:
            findings.append({'sign': 'spo2', 'value': str(spo2), 'status': 'low', 'note': 'Mild hypoxia — monitor closely'})
        else:
            findings.append({'sign': 'spo2', 'value': str(spo2), 'status': 'normal', 'note': 'Oxygen saturation adequate'})

    abnormal = [f for f in findings if f['status'] not in ('normal',)]
    overall = 'critical' if any(f['status'].startswith('critical') for f in findings) \
        else ('abnormal' if abnormal else 'normal')

    _out_str(json.dumps({
        'patient_id': data.get('patient_id', 'unknown'),
        'overall_status': overall,
        'abnormal_count': len(abnormal),
        'findings': findings,
    }))


def action_predict_readmission_risk() -> None:
    """Single patient: predict 30-day hospital readmission risk based on clinical factors."""
    data = _parse_patient_class()

    score = 0.0
    factors: List[str] = []

    age = int(data.get('age', 0))
    if age >= 75:
        score += 20
        factors.append('Age >= 75 (high readmission correlation)')
    elif age >= 65:
        score += 10
        factors.append('Age 65–74')

    history = data.get('medical_history', [])
    if isinstance(history, str):
        history = [h for h in history.split('|') if h]
    chronic_conditions = {'diabetes', 'hypertension', 'smoking'}
    condition_count = sum(1 for c in history if c in chronic_conditions)
    if condition_count >= 3:
        score += 25
        factors.append('Multiple comorbidities (3+)')
    elif condition_count == 2:
        score += 15
        factors.append('Two comorbidities')
    elif condition_count == 1:
        score += 8
        factors.append('One chronic condition')

    cvd = _analyze_cvd(data)
    if cvd['risk_level'] == 'high':
        score += 20
        factors.append('High cardiovascular risk')
    elif cvd['risk_level'] == 'moderate':
        score += 10
        factors.append('Moderate cardiovascular risk')

    dia = _analyze_diabetes(data)
    labs = data.get('lab_results', {})
    hba1c = float(labs.get('hba1c', 0))
    if hba1c >= 8.0:
        score += 15
        factors.append('Poorly controlled diabetes (HbA1c >= 8.0%)')
    elif hba1c >= 6.5:
        score += 8
        factors.append('Elevated HbA1c')

    vitals = data.get('vital_signs', {})
    spo2 = float(vitals.get('spo2', 98))
    if spo2 < 90:
        score += 20
        factors.append('Critical SpO2 — high readmission risk')
    elif spo2 < 95:
        score += 10
        factors.append('Low SpO2')

    score = round(min(score, 100.0), 2)
    level = 'high' if score >= 50 else ('moderate' if score >= 25 else 'low')

    recommendations: List[str] = []
    if level == 'high':
        recommendations.append('Schedule follow-up within 7 days of discharge')
        recommendations.append('Ensure patient has access to a care coordinator')
    if condition_count >= 2:
        recommendations.append('Comprehensive medication reconciliation before discharge')
    recommendations.append('Patient education on warning signs')
    recommendations.append('Confirm post-discharge support system')

    _out_str(json.dumps({
        'patient_id': data.get('patient_id', 'unknown'),
        'readmission_risk_score': score,
        'readmission_risk_level': level,
        'risk_factors': factors,
        'recommendations': recommendations,
    }))


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
    'batch_diabetes_from_file':       action_batch_diabetes_from_file,
    'batch_triage_from_file':         action_batch_triage_from_file,
    'compute_cohort_statistics':      action_compute_cohort_statistics,
    'filter_by_condition':            action_filter_by_condition,
    'compute_risk_distribution':      action_compute_risk_distribution,
    'compute_mortality_risk':         action_compute_mortality_risk,
    'check_vital_signs':              action_check_vital_signs,
    'predict_readmission_risk':       action_predict_readmission_risk,
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