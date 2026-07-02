# Statistics Package

The `statistics` package provides generic CSV analytics for Brane workflows.
It works on registered `Data` inputs and either returns JSON strings or writes
derived CSV/JSON files to `/result/` as `IntermediateResult`s.

## CSV requirements

- First row must be a header row.
- Functions that expect numeric columns ignore empty/non-numeric cells.
- Grouping/counting functions work with any text column.
- If your dataset includes an identifier column such as `patient_id` or `id`,
  outlier results will include it automatically.

## Functions

### `compute_summary_stats(data_file, column)`
Returns count, mean, std_dev, min, max, median, q25, and q75 for one numeric column.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let stats := compute_summary_stats(patients, "glucose");
println(stats);
```

### `count_by_category(data_file, column)`
Returns counts for each unique categorical value.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let counts := count_by_category(patients, "gender");
println(counts);
```

### `compute_correlation(data_file, col_a, col_b)`
Returns Pearson correlation plus a weak/moderate/strong interpretation.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let corr := compute_correlation(patients, "age", "blood_pressure");
println(corr);
```

### `detect_outliers(data_file, column, method)`
Supports `iqr` and `zscore`.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let outliers := detect_outliers(patients, "glucose", "iqr");
println(outliers);
```

### `filter_by_threshold(data_file, column, operator, threshold)`
Writes `/result/filtered.csv` and `/result/filter_summary.json`.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let filtered := filter_by_threshold(patients, "spo2", "lt", 95.0);
commit_result("low_spo2_patients", filtered);
```

### `sort_and_rank(data_file, column, descending)`
Writes `/result/sorted.csv` with an added `rank` column.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let ranked := sort_and_rank(patients, "blood_pressure", "true");
commit_result("bp_ranking", ranked);
```

### `aggregate_by_group(data_file, group_col, value_col, func)`
Supports `mean`, `sum`, `count`, `min`, and `max`.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let avg_by_gender := aggregate_by_group(patients, "gender", "blood_pressure", "mean");
println(avg_by_gender);
```

### `normalize_column(data_file, column, method)`
Supports `minmax` and `zscore`, writing `/result/normalized.csv`.

```brane
import statistics;

let patients := new Data{ name := "patients_stats" };
let normalized := normalize_column(patients, "total_cholesterol", "minmax");
commit_result("cholesterol_normalized", normalized);
```
