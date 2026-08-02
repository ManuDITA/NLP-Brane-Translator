# Epidemics Package

Brane package for population-level epidemic surveillance and analysis.
Provides outbreak detection, reproduction number estimation, stage classification,
incidence rates, and cross-package cohort analysis from healthcare results.

---

## How Brane invokes this package

Same convention as all other packages:
- **Action dispatch** — action name in `argv[1]` via `command.args`
- **Data inputs** — path passed via uppercase env var (e.g. `CASES_FILE`)
- **IntermediateResult inputs** — directory path via uppercase env var (e.g. `ANALYSIS`)
- **String outputs** — `output: "<json-string>"` on stdout
- **Class outputs** — `output: ["EpidemicStatus", {fields}]` (YAML 2-element list)
- **IntermediateResult outputs** — files written to `/result/`

---

## Custom classes

### `EpidemicStatus`

| Field | Type | Description |
|---|---|---|
| `location` | string | Location identifier |
| `stage` | string | `emerging` / `growing` / `plateau` / `declining` / `resolved` |
| `alert_level` | string | `green` / `yellow` / `orange` / `red` |
| `reproduction_number` | real | Estimated Rt (>1 = growing) |
| `peak_daily_cases` | integer | Highest single-day case count |
| `total_cases` | integer | Cumulative total |

**BraneScript usage:**
```
import epidemics;
let cases := new Data{ name := "epidemic_cases" };
let s := get_epidemic_status(cases);
println(s.stage);
if (s.alert_level == "red") {
    println("Emergency response required");
}
```

---

## Functions

### Cases-based functions (take `Data` pointing to a cases CSV)

#### `compute_incidence_rate(cases_file, population)`

Computes cumulative incidence per 100,000 and case-fatality rate per location.

**Returns** JSON string:
```json
{
  "locations": {
    "district_a": {
      "total_cases": 463, "deaths": 11,
      "cumulative_incidence_per_100k": 463.0,
      "case_fatality_rate_pct": 2.38,
      "population": 100000
    }
  },
  "period_days": 20
}
```

---

#### `detect_outbreak(cases_file, threshold)`

Compares recent 3-day average against a rolling baseline. `threshold` is the
multiplier above baseline that triggers an outbreak flag (default `2.0`).

**Returns** JSON string per location with `outbreak_detected`, `alert_level`, `reproduction_number`.

---

#### `estimate_reproduction_number(cases_file)`

Estimates instantaneous Rt using the serial-interval ratio method (window = 5 days).

- Rt > 1: epidemic is growing  
- Rt = 1: stable  
- Rt < 1: declining  

**Returns** JSON string with `rt`, `interpretation`, `epidemic_stage` per location.

---

#### `classify_epidemic_stage(cases_file)`

Classifies the epidemic stage for each location based on the shape of the case curve.

| Stage | Meaning |
|---|---|
| `emerging` | Early rapid growth |
| `growing` | Sustained increase |
| `plateau` | Near peak, stable high |
| `declining` | Falling from peak |
| `resolved` | < 10% of peak, cases minimal |

---

#### `get_epidemic_status(cases_file)` → EpidemicStatus

Returns an `EpidemicStatus` class instance for the most-active location.

---

#### `generate_epidemic_report(cases_file, population)` → IntermediateResult

Full surveillance report. Output files in `/result/`:
- `epidemic_report.json` — complete structured data for all locations
- `location_summary.csv` — tabular summary (compatible with the **statistics** package)
- `summary.html` — colour-coded HTML table

---

### Scalar functions

#### `compute_attack_rate(exposed, cases)`

| Input | Type | Description |
|---|---|---|
| `exposed` | integer | Total population exposed |
| `cases` | integer | Confirmed infections |

**Returns** JSON: `{ "attack_rate_pct": 8.4, "severity": "moderate", ... }`

Severity thresholds: `low` (< 5%), `moderate` (5–9.9%), `high` (10–19.9%), `very_high` (≥ 20%).

---

### Cross-package functions

#### `compute_risk_factor_prevalence(patients_file)`

Accepts the **same patients CSV** as `healthcare.analyze_patients_file()`.
Computes condition prevalence (diabetes, hypertension, smoking), age distribution,
and a `high_cvd_risk_pct` indicator.

**BraneScript usage:**
```
import epidemics;
let patients := new Data{ name := "heal_pa_2" };
let prev := compute_risk_factor_prevalence(patients);
println(prev);
```

---

#### `analyze_health_cohort(analysis)` — IntermediateResult from healthcare

Reads the `IntermediateResult` produced by any of:
- `healthcare.analyze_patients_file()` → CVD analysis
- `healthcare.batch_diabetes_from_file()` → diabetes analysis
- `healthcare.batch_triage_from_file()` → triage analysis

Produces population-level epidemiological indicators: risk distribution,
`population_at_risk_pct`, and a `public_health_alert` level (`low`/`moderate`/`high`).

**BraneScript cross-package workflow:**
```
import healthcare;
import epidemics;

let patients := new Data{ name := "heal_pa_2" };
let analysis := analyze_patients_file(patients);
let cohort   := analyze_health_cohort(analysis);
println(cohort);
```

---

## Cases CSV format

```
date,location,new_cases,total_cases,deaths,recovered,hospitalizations
2026-01-01,district_a,5,5,0,0,2
2026-01-02,district_a,8,13,0,3,5
```

Required columns: `date`, `location`, `new_cases`, `total_cases`.  
Optional columns: `deaths`, `recovered`, `hospitalizations`.  
Multiple locations can coexist in the same file — functions aggregate per location.

---

## Cross-package chaining examples

### Epidemics → Statistics

The `generate_epidemic_report` function writes `location_summary.csv` to `/result/`,
which the **statistics** package can then analyse:

```
import epidemics;
import statistics;

let cases  := new Data{ name := "epidemic_cases" };
let report := generate_epidemic_report(cases, 100000);

# After committing, use statistics on the location_summary.csv
let summary := new Data{ name := "epidemic_cases" };
let rt_dist := compute_summary_stats(summary, "new_cases");
println(rt_dist);

commit_result("epidemic_report", report);
```

### Healthcare → Epidemics

```
import healthcare;
import epidemics;

let patients := new Data{ name := "heal_pa_2" };
let diabetes := batch_diabetes_from_file(patients);
let cohort   := analyze_health_cohort(diabetes);
println(cohort);
```

### Healthcare + Epidemics + Statistics

```
import healthcare;
import epidemics;
import statistics;

let patients := new Data{ name := "heal_pa_2" };
let analysis := analyze_patients_file(patients);

let cohort   := analyze_health_cohort(analysis);
println(cohort);

let cases    := new Data{ name := "epidemic_cases" };
let rt       := estimate_reproduction_number(cases);
println(rt);

let bp_stats := compute_summary_stats(patients, "blood_pressure");
println(bp_stats);
```
