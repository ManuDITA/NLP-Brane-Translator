#!/usr/bin/env python3
"""
Epidemics Package for NLP-Brane-Translator

Population-level epidemic analysis: outbreak detection, reproduction number
estimation, epidemic stage classification, incidence rates, attack rates,
and cross-package cohort analysis from healthcare results.

Brane input convention:
  - Data inputs:              path via uppercase env var (e.g. CASES_FILE)
  - IntermediateResult inputs: directory path via uppercase env var (e.g. ANALYSIS)
  - string/int/real inputs:   JSON-encoded via uppercase env var
  - String outputs:           printed as  output: "<json-string>"
  - Class outputs:            printed as  output: ["ClassName", {fields}]
  - IntermediateResult outputs: files written to /result/

Cross-package compatibility:
  - analyze_health_cohort()       reads IntermediateResult from healthcare
  - compute_risk_factor_prevalence() accepts the same patients CSV as healthcare
  - generate_epidemic_report()    writes CSVs that the statistics package can process
"""

import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ---------------------------------------------------------------------------
# Env var helpers  (same pattern as healthcare/statistics packages)
# ---------------------------------------------------------------------------

def _resolve_env_path(var_name: str) -> str:
    raw = os.environ.get(var_name, '')
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            return decoded
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _env_str(name: str) -> str:
    raw = os.environ.get(name.upper(), '""')
    try:
        v = json.loads(raw)
        return str(v)
    except (json.JSONDecodeError, ValueError):
        return raw


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name.upper(), str(default))
    try:
        return int(json.loads(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name.upper(), str(default))
    try:
        return float(json.loads(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def _out_str(value: str) -> None:
    print(f'output: {json.dumps(str(value))}', flush=True)


def _out_class(class_name: str, fields: Dict[str, Any]) -> None:
    print(yaml.dump({'output': [class_name, fields]}), end='', flush=True)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_cases_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Load a cases time-series CSV. Returns (fieldnames, rows)."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _load_patients_csv(path: str) -> List[Dict[str, Any]]:
    """Load the healthcare-format patients CSV."""
    patients = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history = [h for h in row.get('medical_history', '').split('|') if h]
            patients.append({
                'patient_id': row.get('patient_id', ''),
                'age': int(row.get('age', 0) or 0),
                'gender': row.get('gender', 'O'),
                'medical_history': history,
                'blood_pressure': int(row.get('blood_pressure', 0) or 0),
                'heart_rate': int(row.get('heart_rate', 0) or 0),
                'glucose': float(row.get('glucose', 0) or 0),
            })
    return patients


def _parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core epidemiology logic
# ---------------------------------------------------------------------------

def _group_by_location(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group case rows by location."""
    groups: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        loc = row.get('location', 'unknown')
        groups.setdefault(loc, []).append(row)
    return groups


def _compute_rt(new_cases: List[int], serial_interval: int = 5) -> float:
    """
    Estimate instantaneous reproduction number Rt using the ratio method.
    Compares the sum of cases in two consecutive windows of length serial_interval.
    Returns Rt averaged over all valid windows.
    """
    if len(new_cases) < serial_interval * 2:
        # Not enough data — use simple ratio of last two non-zero values
        non_zero = [v for v in new_cases if v > 0]
        if len(non_zero) < 2:
            return 1.0
        return round(non_zero[-1] / non_zero[-2], 3)

    rt_values = []
    for i in range(serial_interval, len(new_cases) - serial_interval + 1):
        window_past = sum(new_cases[i - serial_interval:i])
        window_curr = sum(new_cases[i:i + serial_interval])
        if window_past > 0:
            rt_values.append(window_curr / window_past)

    return round(sum(rt_values) / len(rt_values), 3) if rt_values else 1.0


def _classify_stage(new_cases: List[int]) -> str:
    """Classify epidemic stage from a daily-case time series."""
    if not new_cases or max(new_cases) == 0:
        return 'resolved'
    if len(new_cases) < 4:
        return 'emerging'

    peak_idx = new_cases.index(max(new_cases))
    last = new_cases[-1]
    peak = max(new_cases)
    recent_avg = sum(new_cases[-3:]) / 3

    if peak_idx < len(new_cases) * 0.25:
        return 'emerging'
    if peak_idx < len(new_cases) * 0.55:
        if recent_avg > peak * 0.8:
            return 'plateau'
        return 'growing'
    if last <= peak * 0.1:
        return 'resolved'
    if last <= peak * 0.4:
        return 'declining'
    return 'plateau'


def _alert_level(rt: float, latest_cases: int) -> str:
    """Map Rt and current case load to a colour-coded alert level."""
    if rt >= 2.0 or latest_cases > 100:
        return 'red'
    if rt >= 1.5 or latest_cases > 40:
        return 'orange'
    if rt >= 1.0 or latest_cases > 10:
        return 'yellow'
    return 'green'


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def action_compute_incidence_rate() -> None:
    """
    Compute cumulative and period incidence rates per 100,000 population
    for each location in a cases CSV dataset.
    """
    path = _resolve_env_path('CASES_FILE')
    population = _env_int('POPULATION', default=100000)
    if not path:
        _out_str(json.dumps({'error': 'CASES_FILE not set'}))
        return

    _, rows = _load_cases_csv(path)
    groups = _group_by_location(rows)
    results = {}

    for loc, loc_rows in groups.items():
        total_cases = sum(int(r.get('total_cases', 0) or 0) for r in loc_rows[-1:])
        new_cases_sum = sum(int(r.get('new_cases', 0) or 0) for r in loc_rows)
        deaths = sum(int(r.get('deaths', 0) or 0) for r in loc_rows)
        cum_incidence = round(total_cases / population * 100_000, 2)
        case_fatality = round(deaths / total_cases * 100, 2) if total_cases > 0 else 0.0
        results[loc] = {
            'total_cases': total_cases,
            'new_cases_period': new_cases_sum,
            'deaths': deaths,
            'cumulative_incidence_per_100k': cum_incidence,
            'case_fatality_rate_pct': case_fatality,
            'population': population,
        }

    _out_str(json.dumps({
        'locations': results,
        'period_days': len(set(r.get('date', '') for r in rows)),
    }))


def action_detect_outbreak() -> None:
    """
    Detect an outbreak in cases data by comparing recent case counts against
    a rolling baseline. Returns alert level and outbreak flag per location.

    threshold: multiplier above baseline that triggers an outbreak (default 2.0)
    """
    path = _resolve_env_path('CASES_FILE')
    threshold = _env_float('THRESHOLD', default=2.0)
    if not path:
        _out_str(json.dumps({'error': 'CASES_FILE not set'}))
        return

    _, rows = _load_cases_csv(path)
    groups = _group_by_location(rows)
    results = {}

    for loc, loc_rows in groups.items():
        case_counts = [int(r.get('new_cases', 0) or 0) for r in loc_rows]
        if len(case_counts) < 4:
            results[loc] = {'outbreak_detected': False, 'alert': 'insufficient_data'}
            continue

        # Baseline = mean of first half; recent = mean of last 3 days
        mid = max(len(case_counts) // 2, 2)
        baseline = sum(case_counts[:mid]) / mid
        recent_avg = sum(case_counts[-3:]) / 3
        peak = max(case_counts)
        peak_date = loc_rows[case_counts.index(peak)].get('date', 'unknown')
        outbreak = recent_avg > baseline * threshold

        rt = _compute_rt(case_counts)
        results[loc] = {
            'outbreak_detected': outbreak,
            'baseline_daily_avg': round(baseline, 2),
            'recent_3day_avg': round(recent_avg, 2),
            'threshold_multiplier': threshold,
            'peak_daily_cases': peak,
            'peak_date': peak_date,
            'reproduction_number': rt,
            'alert_level': _alert_level(rt, int(case_counts[-1])),
        }

    _out_str(json.dumps({
        'locations': results,
    }))


def action_estimate_reproduction_number() -> None:
    """
    Estimate the instantaneous reproduction number Rt for each location
    in a cases CSV dataset using the serial interval ratio method.

    Rt > 1: epidemic is growing
    Rt = 1: epidemic is stable
    Rt < 1: epidemic is declining
    """
    path = _resolve_env_path('CASES_FILE')
    if not path:
        _out_str(json.dumps({'error': 'CASES_FILE not set'}))
        return

    _, rows = _load_cases_csv(path)
    groups = _group_by_location(rows)
    results = {}

    for loc, loc_rows in groups.items():
        case_counts = [int(r.get('new_cases', 0) or 0) for r in loc_rows]
        rt = _compute_rt(case_counts)
        stage = _classify_stage(case_counts)
        interpretation = (
            'epidemic growing' if rt > 1.0
            else ('epidemic stable' if rt == 1.0 else 'epidemic declining')
        )
        results[loc] = {
            'rt': rt,
            'interpretation': interpretation,
            'epidemic_stage': stage,
            'total_days': len(case_counts),
            'latest_daily_cases': case_counts[-1] if case_counts else 0,
        }

    _out_str(json.dumps({
        'locations': results,
    }))


def action_classify_epidemic_stage() -> None:
    """
    Classify the epidemic stage for each location as:
    emerging / growing / plateau / declining / resolved.
    """
    path = _resolve_env_path('CASES_FILE')
    if not path:
        _out_str(json.dumps({'error': 'CASES_FILE not set'}))
        return

    _, rows = _load_cases_csv(path)
    groups = _group_by_location(rows)
    results = {}

    for loc, loc_rows in groups.items():
        case_counts = [int(r.get('new_cases', 0) or 0) for r in loc_rows]
        stage = _classify_stage(case_counts)
        peak = max(case_counts) if case_counts else 0
        total = int(loc_rows[-1].get('total_cases', 0) or 0) if loc_rows else 0
        rt = _compute_rt(case_counts)
        results[loc] = {
            'stage': stage,
            'alert_level': _alert_level(rt, case_counts[-1] if case_counts else 0),
            'peak_daily_cases': peak,
            'total_cases': total,
            'reproduction_number': rt,
        }

    _out_str(json.dumps({
        'locations': results,
    }))


def action_compute_attack_rate() -> None:
    """
    Compute attack rate (proportion of exposed population that got infected).
    Inputs: exposed (integer), cases (integer).
    Returns attack rate %, severity classification, and herd immunity threshold.
    """
    exposed = _env_int('EXPOSED', default=1)
    cases = _env_int('CASES', default=0)

    if exposed <= 0:
        _out_str(json.dumps({'error': 'exposed must be > 0'}))
        return

    attack_rate = round(cases / exposed * 100, 2)
    if attack_rate >= 20:
        severity = 'very_high'
    elif attack_rate >= 10:
        severity = 'high'
    elif attack_rate >= 5:
        severity = 'moderate'
    else:
        severity = 'low'

    _out_class('AttackRate', {
        'attack_rate_pct': attack_rate,
        'severity': severity,
        'exposed': exposed,
        'cases': cases,
    })


def action_compute_risk_factor_prevalence() -> None:
    """
    Compute prevalence of each risk factor (medical_history condition) in the
    patient population. Accepts the same patients CSV format as the healthcare package.
    """
    path = _resolve_env_path('PATIENTS_FILE')
    if not path:
        _out_str(json.dumps({'error': 'PATIENTS_FILE not set'}))
        return

    patients = _load_patients_csv(path)
    if not patients:
        _out_str(json.dumps({'error': 'No patient records found'}))
        return

    total = len(patients)
    condition_counts: Dict[str, int] = {}
    age_groups: Dict[str, int] = {'0-17': 0, '18-44': 0, '45-64': 0, '65+': 0}
    gender_counts: Dict[str, int] = {}

    for p in patients:
        for cond in p['medical_history']:
            condition_counts[cond] = condition_counts.get(cond, 0) + 1
        age = p['age']
        if age < 18:
            age_groups['0-17'] += 1
        elif age < 45:
            age_groups['18-44'] += 1
        elif age < 65:
            age_groups['45-64'] += 1
        else:
            age_groups['65+'] += 1
        g = p.get('gender', 'O').upper()
        gender_counts[g] = gender_counts.get(g, 0) + 1

    prevalence = {
        cond: {
            'count': cnt,
            'prevalence_pct': round(cnt / total * 100, 1),
        }
        for cond, cnt in sorted(condition_counts.items(), key=lambda x: -x[1])
    }
    age_distribution = {
        grp: {'count': cnt, 'pct': round(cnt / total * 100, 1)}
        for grp, cnt in age_groups.items()
    }

    # Estimate population-level cardiovascular risk:
    # patients with hypertension or smoking are high-risk markers
    high_cvd_markers = sum(
        1 for p in patients
        if any(c in ('hypertension', 'smoking') for c in p['medical_history'])
    )

    _out_str(json.dumps({
        'total_patients': total,
        'condition_prevalence': prevalence,
        'age_distribution': age_distribution,
        'gender_distribution': gender_counts,
        'high_cvd_risk_pct': round(high_cvd_markers / total * 100, 1),
    }))


def action_analyze_health_cohort() -> None:
    """
    Cross-package function: reads an IntermediateResult produced by
    healthcare.analyze_patients_file(), healthcare.batch_diabetes_from_file(),
    or healthcare.batch_triage_from_file() and produces population-level
    epidemiological indicators.
    """
    result_dir = _resolve_env_path('ANALYSIS')
    if not result_dir:
        _out_str(json.dumps({'error': 'ANALYSIS not set — pass an IntermediateResult from healthcare'}))
        return

    # Accept any of the three healthcare batch report types
    report_path = None
    report_type = None
    for fname, rtype in [
        ('analysis_report.json', 'cvd'),
        ('diabetes_report.json', 'diabetes'),
        ('triage_report.json', 'triage'),
    ]:
        for root, _, files in os.walk(result_dir):
            if fname in files:
                report_path = os.path.join(root, fname)
                report_type = rtype
                break
        if report_path:
            break

    if report_path is None:
        _out_str(json.dumps({'error': f'No recognised healthcare report found under {result_dir}'}))
        return

    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = data.get('total_records', 0)
    results = data.get('results', [])

    if report_type in ('cvd', 'diabetes'):
        dist: Dict[str, int] = {'low': 0, 'moderate': 0, 'high': 0}
        scores = []
        for entry in results:
            level = entry.get('risk_level', 'low')
            dist[level] = dist.get(level, 0) + 1
            score = entry.get('risk_score')
            if score is not None:
                scores.append(float(score))

        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
        high_pct = round(dist.get('high', 0) / total * 100, 1) if total > 0 else 0.0
        # Heuristic: population attack rate proxy — high risk % as at-risk fraction
        population_at_risk_pct = round(
            (dist.get('high', 0) + dist.get('moderate', 0)) / total * 100, 1
        ) if total > 0 else 0.0

        _out_str(json.dumps({
            'analysis_type': report_type,
            'total_patients': total,
            'risk_distribution': dist,
            'high_risk_pct': high_pct,
            'population_at_risk_pct': population_at_risk_pct,
            'average_risk_score': avg_score,
            'public_health_alert': 'high' if high_pct >= 30 else ('moderate' if high_pct >= 15 else 'low'),
        }))

    else:  # triage
        patients_list = data.get('patients', [])
        summary = data.get('summary', {})
        immediate_pct = round(summary.get('immediate', 0) / total * 100, 1) if total > 0 else 0.0
        _out_str(json.dumps({
            'analysis_type': 'triage',
            'total_patients': total,
            'triage_distribution': summary,
            'immediate_care_pct': immediate_pct,
            'public_health_alert': 'high' if immediate_pct >= 20 else ('moderate' if immediate_pct >= 5 else 'low'),
        }))


def action_get_epidemic_status() -> None:
    """
    Return an EpidemicStatus class instance for the most active location
    in a cases CSV. Class fields are directly accessible in BraneScript.

    BraneScript usage:
        let status := get_epidemic_status(cases);
        println(status.stage);
        println(status.alert_level);
    """
    path = _resolve_env_path('CASES_FILE')
    if not path:
        _out_str(json.dumps({'error': 'CASES_FILE not set'}))
        return

    _, rows = _load_cases_csv(path)
    groups = _group_by_location(rows)

    # Pick the location with the highest total cases
    best_loc = max(
        groups.keys(),
        key=lambda loc: int(groups[loc][-1].get('total_cases', 0) or 0)
    )
    loc_rows = groups[best_loc]
    case_counts = [int(r.get('new_cases', 0) or 0) for r in loc_rows]
    total_cases = int(loc_rows[-1].get('total_cases', 0) or 0)
    peak = max(case_counts) if case_counts else 0

    rt = _compute_rt(case_counts)
    stage = _classify_stage(case_counts)
    alert = _alert_level(rt, case_counts[-1] if case_counts else 0)

    _out_class('EpidemicStatus', {
        'location': best_loc,
        'stage': stage,
        'alert_level': alert,
        'reproduction_number': float(rt),
        'peak_daily_cases': int(peak),
        'total_cases': int(total_cases),
    })


def action_generate_epidemic_report() -> None:
    """
    Full epidemic report: combines incidence rate, outbreak detection, Rt,
    and stage classification for all locations. Writes to /result/:
      - epidemic_report.json    full structured report
      - location_summary.csv    per-location summary table (for statistics package)
      - summary.html            human-readable HTML table
    """
    path = _resolve_env_path('CASES_FILE')
    population = _env_int('POPULATION', default=100000)
    if not path:
        raise RuntimeError('CASES_FILE not set')

    _, rows = _load_cases_csv(path)
    groups = _group_by_location(rows)

    locations_report = {}
    csv_rows = []

    for loc, loc_rows in groups.items():
        case_counts = [int(r.get('new_cases', 0) or 0) for r in loc_rows]
        total = int(loc_rows[-1].get('total_cases', 0) or 0)
        deaths = sum(int(r.get('deaths', 0) or 0) for r in loc_rows)
        rt = _compute_rt(case_counts)
        stage = _classify_stage(case_counts)
        alert = _alert_level(rt, case_counts[-1] if case_counts else 0)
        cum_incidence = round(total / population * 100_000, 2)
        cfr = round(deaths / total * 100, 2) if total > 0 else 0.0
        peak = max(case_counts) if case_counts else 0
        peak_date = loc_rows[case_counts.index(peak)].get('date', 'unknown') if peak else 'unknown'

        loc_entry = {
            'location': loc,
            'total_cases': total,
            'deaths': deaths,
            'case_fatality_rate_pct': cfr,
            'cumulative_incidence_per_100k': cum_incidence,
            'reproduction_number': rt,
            'epidemic_stage': stage,
            'alert_level': alert,
            'peak_daily_cases': peak,
            'peak_date': peak_date,
            'days_observed': len(loc_rows),
        }
        locations_report[loc] = loc_entry
        csv_rows.append({
            'location': loc,
            'total_cases': total,
            'deaths': deaths,
            'cfr_pct': cfr,
            'incidence_per_100k': cum_incidence,
            'rt': rt,
            'stage': stage,
            'alert_level': alert,
            'peak_daily_cases': peak,
        })

    total_cases_all = sum(v['total_cases'] for v in locations_report.values())
    deaths_all = sum(v['deaths'] for v in locations_report.values())
    active_locs = [loc for loc, v in locations_report.items() if v['epidemic_stage'] not in ('resolved',)]

    report = {
        'population': population,
        'total_cases_all_locations': total_cases_all,
        'total_deaths_all_locations': deaths_all,
        'active_locations': active_locs,
        'locations': locations_report,
    }

    os.makedirs('/result', exist_ok=True)
    with open('/result/epidemic_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    with open('/result/location_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    rows_html = ''.join(
        f"<tr>"
        f"<td>{e['location']}</td><td>{e['total_cases']}</td>"
        f"<td>{e['cfr_pct']}%</td><td>{e['incidence_per_100k']}</td>"
        f"<td>{e['rt']}</td><td>{e['stage']}</td>"
        f"<td style='color:{'red' if e['alert_level']=='red' else ('orange' if e['alert_level']=='orange' else ('goldenrod' if e['alert_level']=='yellow' else 'green'))}'>"
        f"{e['alert_level'].upper()}</td>"
        f"</tr>"
        for e in csv_rows
    )
    html = (
        '<!DOCTYPE html><html><head><title>Epidemic Report</title></head><body>'
        '<h1>Epidemic Surveillance Report</h1>'
        f'<p>Generated: {ts} | Population: {population:,}</p>'
        f'<p>Total cases: {total_cases_all} | Deaths: {deaths_all} | '
        f'Active locations: {len(active_locs)}</p>'
        '<table border="1" cellpadding="4">'
        '<tr><th>Location</th><th>Total Cases</th><th>CFR</th>'
        '<th>Incidence/100k</th><th>Rt</th><th>Stage</th><th>Alert</th></tr>'
        f'{rows_html}</table></body></html>'
    )
    with open('/result/summary.html', 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ACTIONS = {
    'compute_incidence_rate':        action_compute_incidence_rate,
    'detect_outbreak':               action_detect_outbreak,
    'estimate_reproduction_number':  action_estimate_reproduction_number,
    'classify_epidemic_stage':       action_classify_epidemic_stage,
    'compute_attack_rate':           action_compute_attack_rate,
    'compute_risk_factor_prevalence': action_compute_risk_factor_prevalence,
    'analyze_health_cohort':         action_analyze_health_cohort,
    'get_epidemic_status':           action_get_epidemic_status,
    'generate_epidemic_report':      action_generate_epidemic_report,
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
