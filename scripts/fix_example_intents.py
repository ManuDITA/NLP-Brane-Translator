from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRING_ASSIGN_RE = re.compile(r'let\s+(\w+)\s*:=\s*"((?:\\.|[^"\\])*)";', re.S)
DATA_ASSIGN_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*new\s+Data\{\s*name\s*:=\s*"([^"]+)"\s*\};', re.S
)
PARALLEL_ASSIGN_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*parallel\s*\[all\]\s*\[(.*?)\];', re.S
)
CALL_ASSIGN_RE = re.compile(
    r'let\s+(\w+)\s*:=\s*([A-Za-z_][A-Za-z0-9_]*)\((.*?)\);', re.S
)
PRINT_RE = re.compile(r"println\((.*?)\);", re.S)
COMMIT_RE = re.compile(r'commit_result\("([^"]+)",\s*(.*?)\);', re.S)
RETURN_CALL_RE = re.compile(r"return\s+([A-Za-z_][A-Za-z0-9_]*)\((.*?)\);", re.S)


@dataclass
class VarInfo:
    kind: str
    value: Any
    phrase: str | None = None
    label: str | None = None
    parts: list["VarInfo"] | None = None


def oxford_join(items: list[str]) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def capitalize_first(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def humanize_name(name: str) -> str:
    special = {
        "patient_id": "patient ID",
        "blood_pressure": "blood pressure",
        "heart_rate": "heart rate",
        "total_cholesterol": "cholesterol",
        "hba1c": "HbA1c",
        "spo2": "SpO2",
        "weight_kg": "weight",
        "height_cm": "height",
        "new_cases": "new cases",
        "total_cases": "total cases",
        "peak_daily_cases": "peak daily cases",
        "avg_word_length": "average word length",
        "risk_level": "risk level",
        "risk_score": "risk score",
        "readability_score": "readability score",
        "alert_level": "alert level",
        "reproduction_number": "reproduction number",
    }
    if name in special:
        return special[name]
    return name.replace("_", " ")


def format_number(value: Any) -> str:
    if isinstance(value, float):
        text = str(value)
        if "." in text and text.endswith("0"):
            return f"{value:.1f}"
        return text
    return str(value)


def decode_brane_string(raw: str) -> str:
    return ast.literal_eval(f'"{raw}"')


def parse_literal(token: str) -> Any:
    token = token.strip()
    if not token:
        return ""
    if token.startswith('"') and token.endswith('"'):
        return decode_brane_string(token[1:-1])
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if re.fullmatch(r"-?\d+\.\d+", token):
        return float(token)
    return token


def split_args(arg_string: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    in_string = False
    escaped = False
    depth = 0
    for char in arg_string:
        if in_string:
            current.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def format_history(history: list[str]) -> str:
    if not history:
        return "no known medical history"
    return f"a medical history of {oxford_join(history)}"


def format_patient(patient: dict[str, Any]) -> str:
    vitals = patient.get("vital_signs", {})
    labs = patient.get("lab_results", {})
    gender = {"M": "male", "F": "female"}.get(patient.get("gender"), str(patient.get("gender")).lower())
    return (
        f"a {patient['age']}-year-old {gender} patient ({patient['patient_id']}) "
        f"with blood pressure {format_number(vitals['blood_pressure'])}, "
        f"heart rate {format_number(vitals['heart_rate'])}, "
        f"SpO2 {format_number(vitals['spo2'])}%, "
        f"temperature {format_number(vitals['temperature'])}°C, "
        f"weight {format_number(vitals['weight_kg'])}kg, "
        f"height {format_number(vitals['height_cm'])}cm, "
        f"cholesterol {format_number(labs['total_cholesterol'])}, "
        f"glucose {format_number(labs['glucose'])}, "
        f"HbA1c {format_number(labs['hba1c'])}, "
        f"and {format_history(patient.get('medical_history', []))}"
    )


def format_record(record: dict[str, Any]) -> str:
    parts = [f"{humanize_name(key)} {format_number(value)}" for key, value in record.items()]
    return f"the record with {oxford_join(parts)}"


def format_fields(fields: list[str]) -> str:
    human = [humanize_name(field) for field in fields]
    if len(human) == 1:
        return f"the {human[0]} field"
    return f"the {oxford_join(human)} fields"


def pretty_format(fmt: str) -> str:
    mapping = {
        "%Y-%m-%d": "YYYY-MM-DD",
        "%Y-%m-%d %H:%M": "YYYY-MM-DD HH:MM",
        "%d/%m/%Y": "DD/MM/YYYY",
        "%d-%m-%Y": "DD-MM-YYYY",
        "%Y/%m/%d": "YYYY/MM/DD",
        "%B %Y": "Month YYYY",
        "%Y-%m-%dT%H:%M:%S": "YYYY-MM-DDTHH:MM:SS",
    }
    return mapping.get(fmt, fmt)


class IntentGenerator:
    def __init__(self, branescript: str):
        self.branescript = branescript
        self.variables: dict[str, VarInfo] = {}
        self.statements: list[tuple[int, int, str, Any]] = []
        self.last_patient_var: str | None = None
        self._collect_statements()

    def _collect_statements(self) -> None:
        occupied: list[tuple[int, int]] = []

        def add(kind: str, match: re.Match[str], payload: Any) -> None:
            self.statements.append((match.start(), match.end(), kind, payload))
            occupied.append((match.start(), match.end()))

        for match in PARALLEL_ASSIGN_RE.finditer(self.branescript):
            add("parallel", match, match.groups())

        def is_inside_occupied(start: int, end: int) -> bool:
            return any(start >= s and end <= e for s, e in occupied)

        for regex, kind in [
            (STRING_ASSIGN_RE, "string"),
            (DATA_ASSIGN_RE, "data"),
            (CALL_ASSIGN_RE, "call"),
            (PRINT_RE, "print"),
            (COMMIT_RE, "commit"),
        ]:
            for match in regex.finditer(self.branescript):
                if is_inside_occupied(match.start(), match.end()):
                    continue
                add(kind, match, match.groups())

        self.statements.sort(key=lambda item: item[0])

    def generate(self) -> str:
        phrases: list[str] = []
        i = 0
        while i < len(self.statements):
            _, _, kind, payload = self.statements[i]
            if kind == "string":
                self._handle_string_assignment(*payload)
                i += 1
                continue
            if kind == "data":
                self._handle_data_assignment(*payload)
                i += 1
                continue
            if kind == "call":
                phrases.append(self._handle_call_assignment(*payload))
                i += 1
                continue
            if kind == "parallel":
                phrases.append(self._handle_parallel_assignment(*payload))
                i += 1
                continue
            if kind == "print":
                print_exprs: list[str] = []
                while i < len(self.statements) and self.statements[i][2] == "print":
                    print_exprs.append(self.statements[i][3][0])
                    i += 1
                phrases.append(self._describe_prints(print_exprs))
                continue
            if kind == "commit":
                commits: list[tuple[str, str]] = []
                while i < len(self.statements) and self.statements[i][2] == "commit":
                    name, expr = self.statements[i][3]
                    commits.append((name, expr))
                    i += 1
                phrases.append(self._describe_commits(commits))
                continue
        if not phrases:
            raise ValueError("No actionable statements found")
        if len(phrases) == 1:
            return capitalize_first(phrases[0])
        if len(phrases) == 2:
            return f"{capitalize_first(phrases[0])}, and {phrases[1]}"
        return f"{capitalize_first(phrases[0])}, {', '.join(phrases[1:-1])}, and {phrases[-1]}"

    def _handle_string_assignment(self, var_name: str, raw_value: str) -> None:
        value = decode_brane_string(raw_value)
        phrase = None
        label = None
        kind = "text"
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            if isinstance(parsed, dict) and {"patient_id", "age", "gender"} <= parsed.keys():
                kind = "patient"
                phrase = format_patient(parsed)
                label = "the patient"
                self.variables[var_name] = VarInfo(kind, parsed, phrase, label)
                return
            if isinstance(parsed, dict):
                kind = "record"
                phrase = format_record(parsed)
                label = "the record"
                self.variables[var_name] = VarInfo(kind, parsed, phrase, label)
                return
            if isinstance(parsed, list):
                kind = "list"
                phrase = format_fields(parsed)
                label = phrase
                self.variables[var_name] = VarInfo(kind, parsed, phrase, label)
                return
        phrase = f"the text '{value}'"
        label = phrase
        self.variables[var_name] = VarInfo(kind, value, phrase, label)

    def _handle_data_assignment(self, var_name: str, dataset_name: str) -> None:
        self.variables[var_name] = VarInfo(
            kind="data",
            value=dataset_name,
            phrase=f"the {dataset_name} dataset",
            label=f"the {dataset_name} dataset",
        )

    def _handle_call_assignment(self, var_name: str, func_name: str, arg_string: str) -> str:
        args = split_args(arg_string)
        phrase, label = self._action_phrase(func_name, args)
        self.variables[var_name] = VarInfo(kind="action", value=(func_name, args), phrase=phrase, label=label)
        self._remember_subject(args)
        return phrase

    def _handle_parallel_assignment(self, var_name: str, block: str) -> str:
        parts: list[VarInfo] = []
        subphrases: list[str] = []
        for func_name, arg_string in RETURN_CALL_RE.findall(block):
            args = split_args(arg_string)
            phrase, label = self._action_phrase(func_name, args)
            parts.append(VarInfo(kind="action", value=(func_name, args), phrase=phrase, label=label))
            subphrases.append(phrase)
        phrase = f"run these in parallel: {oxford_join(subphrases)}"
        self.variables[var_name] = VarInfo(kind="parallel", value=None, phrase=phrase, label="the parallel results", parts=parts)
        for part in parts:
            self._remember_subject(part.value[1])
        return phrase

    def _remember_subject(self, args: list[str]) -> None:
        if not args:
            return
        first = args[0].strip()
        if first in self.variables and self.variables[first].kind == "patient":
            self.last_patient_var = first

    def _subject_text(self, arg: str, allow_same_patient: bool = True) -> str:
        arg = arg.strip()
        if arg in self.variables:
            info = self.variables[arg]
            if info.kind == "patient":
                if allow_same_patient and self.last_patient_var == arg:
                    return "the same patient"
                return info.phrase or info.label or arg
            return info.phrase or info.label or arg
        literal = parse_literal(arg)
        if isinstance(literal, str):
            return literal
        return format_number(literal)

    def _dataset_subject(self, arg: str) -> str:
        info = self.variables[arg.strip()]
        return info.phrase or info.label or arg

    def _describe_source(self, arg: str) -> str:
        arg = arg.strip()
        if match := re.fullmatch(r"(\w+)\[(\d+)\]", arg):
            var_name, index_text = match.groups()
            info = self.variables[var_name]
            if info.parts:
                part = info.parts[int(index_text)]
                return part.label or "the result"
        if arg in self.variables:
            info = self.variables[arg]
            return info.label or info.phrase or arg
        literal = parse_literal(arg)
        if isinstance(literal, str):
            return literal
        return format_number(literal)

    def _describe_value(self, arg: str) -> str:
        arg = arg.strip()
        if arg in self.variables:
            info = self.variables[arg]
            if info.kind == "text":
                return f"the value '{info.value}'"
            return info.label or info.phrase or arg
        literal = parse_literal(arg)
        if isinstance(literal, str):
            return f"the value '{literal}'"
        return format_number(literal)

    def _action_phrase(self, func_name: str, args: list[str]) -> tuple[str, str]:
        if func_name == "analyze_heart_disease":
            patient = self._subject_text(args[0])
            return f"analyze heart disease risk for {patient}", "the heart disease risk analysis"
        if func_name == "assess_diabetes_risk":
            patient = self._subject_text(args[0])
            return f"assess diabetes risk for {patient}", "the diabetes risk assessment"
        if func_name == "generate_report":
            patient = self._subject_text(args[0])
            return f"generate a full health report for {patient}", "the full health report"
        if func_name == "triage_patient":
            patient = self._subject_text(args[0])
            return f"triage {patient}", "the triage result"
        if func_name == "check_vital_signs":
            patient = self._subject_text(args[0])
            return f"check the vital signs for {patient}", "the vital sign assessment"
        if func_name == "get_patient_summary":
            patient = self._subject_text(args[0])
            return f"get a patient summary for {patient}", "the patient summary"
        if func_name == "predict_readmission_risk":
            patient = self._subject_text(args[0])
            return f"predict 30-day hospital readmission risk for {patient}", "the 30-day readmission risk assessment"
        if func_name == "compute_mortality_risk":
            patient = self._subject_text(args[0])
            return f"compute mortality risk for {patient}", "the mortality risk assessment"
        if func_name == "validate_patient_data":
            patient = self._subject_text(args[0])
            return f"validate the data for {patient}", "the validation result"
        if func_name == "compute_bmi":
            weight = format_number(parse_literal(args[0]))
            height = format_number(parse_literal(args[1]))
            return f"compute the BMI for a patient weighing {weight}kg and measuring {height}cm", "the BMI"
        if func_name == "analyze_patients_file":
            dataset = self._dataset_subject(args[0])
            return f"run cardiovascular analysis on all patients in {dataset}", f"the cardiovascular analysis of all patients in {dataset}"
        if func_name == "batch_diabetes_from_file":
            dataset = self._dataset_subject(args[0])
            return f"assess diabetes risk for all patients in {dataset}", f"the diabetes risk assessment for all patients in {dataset}"
        if func_name == "batch_triage_from_file":
            dataset = self._dataset_subject(args[0])
            return f"triage all patients in {dataset}", f"the triage report for all patients in {dataset}"
        if func_name == "compute_cohort_statistics":
            dataset = self._dataset_subject(args[0])
            return f"compute cohort statistics for {dataset}", f"the cohort statistics for {dataset}"
        if func_name == "filter_high_risk_patients":
            dataset = self._dataset_subject(args[0])
            return f"filter all patients in {dataset} to only high-risk patients", "the filtered high-risk patients"
        if func_name == "filter_by_condition":
            dataset = self._dataset_subject(args[0])
            condition = parse_literal(args[1])
            return f"filter all patients in {dataset} to only those with {condition}", f"the filtered patients with {condition}"
        if func_name == "compute_summary_stats":
            dataset = self._dataset_subject(args[0])
            column = humanize_name(parse_literal(args[1]))
            return f"compute summary statistics for the {column} column in {dataset}", f"the summary statistics for the {column} column"
        if func_name == "count_by_category":
            dataset = self._dataset_subject(args[0])
            column = humanize_name(parse_literal(args[1]))
            return f"count the records in {dataset} by {column}", f"the counts by {column}"
        if func_name == "compute_correlation":
            dataset = self._dataset_subject(args[0])
            first = humanize_name(parse_literal(args[1]))
            second = humanize_name(parse_literal(args[2]))
            return f"compute the correlation between {first} and {second} in {dataset}", f"the correlation between {first} and {second}"
        if func_name == "detect_outliers":
            dataset = self._dataset_subject(args[0])
            column = humanize_name(parse_literal(args[1]))
            method = parse_literal(args[2])
            return f"detect outliers in the {column} column of {dataset} using the {method} method", f"the {column} outliers"
        if func_name == "filter_by_threshold":
            dataset = self._dataset_subject(args[0])
            column = humanize_name(parse_literal(args[1]))
            operator = {
                "gt": "greater than",
                "gte": "greater than or equal to",
                "lt": "less than",
                "lte": "less than or equal to",
                "eq": "equal to",
            }.get(parse_literal(args[2]), parse_literal(args[2]))
            threshold = format_number(parse_literal(args[3]))
            return f"filter the records in {dataset} where {column} is {operator} {threshold}", "the filtered records"
        if func_name == "sort_and_rank":
            dataset = self._dataset_subject(args[0])
            column = humanize_name(parse_literal(args[1]))
            direction = "descending" if str(parse_literal(args[2])).lower() == "true" else "ascending"
            return f"sort and rank the records in {dataset} by {column} in {direction} order", f"the records ranked by {column} in {direction} order"
        if func_name == "aggregate_by_group":
            dataset = self._dataset_subject(args[0])
            group = humanize_name(parse_literal(args[1]))
            value = humanize_name(parse_literal(args[2]))
            aggregate = parse_literal(args[3])
            return f"compute the {aggregate} {value} grouped by {group} in {dataset}", f"the {aggregate} {value} grouped by {group}"
        if func_name == "normalize_column":
            dataset = self._dataset_subject(args[0])
            column = humanize_name(parse_literal(args[1]))
            method = parse_literal(args[2])
            return f"normalize the {column} column in {dataset} using {method} normalization", f"the {column} column normalized with {method}"
        if func_name == "compute_incidence_rate":
            dataset = self._dataset_subject(args[0])
            population = format_number(parse_literal(args[1]))
            return f"compute the incidence rate in {dataset} per {population} people", f"the incidence rate per {population} people"
        if func_name == "detect_outbreak":
            dataset = self._dataset_subject(args[0])
            threshold = format_number(parse_literal(args[1]))
            return f"detect outbreaks in {dataset} using a threshold of {threshold}", f"the outbreak detections with a threshold of {threshold}"
        if func_name == "estimate_reproduction_number":
            dataset = self._dataset_subject(args[0])
            return f"estimate the reproduction number for {dataset}", f"the reproduction number for {dataset}"
        if func_name == "classify_epidemic_stage":
            dataset = self._dataset_subject(args[0])
            return f"classify the epidemic stage for {dataset}", f"the epidemic stage classifications for {dataset}"
        if func_name == "get_epidemic_status":
            dataset = self._dataset_subject(args[0])
            return f"get the epidemic status for {dataset}", f"the epidemic status for {dataset}"
        if func_name == "generate_epidemic_report":
            dataset = self._dataset_subject(args[0])
            population = format_number(parse_literal(args[1]))
            return f"generate a full epidemic surveillance report for {dataset} using a population of {population}", f"the epidemic report for {dataset}"
        if func_name == "compute_attack_rate":
            exposed = format_number(parse_literal(args[0]))
            confirmed = format_number(parse_literal(args[1]))
            return f"compute the attack rate for {confirmed} confirmed cases out of {exposed} exposed people", "the attack rate"
        if func_name == "compute_risk_factor_prevalence":
            dataset = self._dataset_subject(args[0])
            return f"compute the prevalence of risk factors in {dataset}", f"the risk factor prevalence in {dataset}"
        if func_name == "analyze_health_cohort":
            source = self._describe_source(args[0])
            return f"analyze the health cohort from {source}", f"the health cohort analysis from {source}"
        if func_name == "compute_risk_distribution":
            source = self._describe_source(args[0])
            return f"compute the risk distribution from {source}", f"the risk distribution from {source}"
        if func_name == "count_words":
            text = self._subject_text(args[0], allow_same_patient=False)
            return f"count the number of words in {text}", "the word count"
        if func_name == "count_sentences":
            text = self._subject_text(args[0], allow_same_patient=False)
            return f"count the number of sentences in {text}", "the sentence count"
        if func_name == "compute_readability":
            text = self._subject_text(args[0], allow_same_patient=False)
            return f"compute the readability score of {text}", "the readability score"
        if func_name == "compute_sentiment":
            text = self._subject_text(args[0], allow_same_patient=False)
            return f"compute the sentiment of {text}", "the sentiment"
        if func_name == "extract_keywords":
            text = self._subject_text(args[0], allow_same_patient=False)
            count = format_number(parse_literal(args[1]))
            return f"extract the top {count} keywords from {text}", "the keywords"
        if func_name == "get_text_stats":
            text = self._subject_text(args[0], allow_same_patient=False)
            return f"get text statistics for {text}", "the text statistics"
        if func_name == "analyze_text_file":
            dataset = self._dataset_subject(args[0])
            return f"analyze the texts in {dataset}", f"the text analysis of {dataset}"
        if func_name == "generate_frequency_report":
            source = self._describe_source(args[0])
            return f"generate a frequency report from {source}", f"the frequency report from {source}"
        if func_name == "detect_pii":
            text = self._subject_text(args[0], allow_same_patient=False)
            return f"detect personally identifiable information in {text}", "the detected personally identifiable information"
        if func_name == "mask_json_record":
            record_arg = args[0].strip()
            if record_arg in self.variables:
                record = self.variables[record_arg].phrase or self.variables[record_arg].label or record_arg
            else:
                record = self._describe_source(args[0])
            fields = self._describe_source(args[1])
            return f"mask {fields} in {record}", "the masked record"
        if func_name == "mask_csv_file":
            dataset = self._dataset_subject(args[0])
            fields = self._describe_source(args[1])
            return f"mask {fields} in {dataset}", f"the masked version of {dataset}"
        if func_name == "generate_masking_report":
            source = self._describe_source(args[0])
            return f"generate a masking report from {source}", f"the masking report from {source}"
        if func_name == "mask_value":
            value = self._describe_value(args[0])
            strategy = parse_literal(args[1])
            return f"mask {value} using the {strategy} strategy", "the masked value"
        if func_name == "get_iso":
            return "get the current date and time in ISO 8601 format", "the ISO 8601 timestamp"
        if func_name == "get_human":
            return "get the current human-readable timestamp", "the human-readable timestamp"
        if func_name == "get_formatted":
            fmt = pretty_format(parse_literal(args[0]))
            return f"get the current date and time formatted as {fmt}", f"the date and time formatted as {fmt}"
        if func_name == "get_date":
            return "get the current date", "the current date"
        if func_name == "get_time":
            return "get the current time", "the current time"
        if func_name == "get_unix":
            return "get the current Unix timestamp", "the Unix timestamp"
        raise ValueError(f"Unsupported function: {func_name}")

    def _describe_expression(self, expr: str) -> str:
        expr = expr.strip()
        if match := re.fullmatch(r"(\w+)\[(\d+)\]", expr):
            var_name, index_text = match.groups()
            info = self.variables[var_name]
            if info.parts and len(info.parts) == 2:
                return f"the {['first', 'second'][int(index_text)]} result"
            if info.parts:
                return info.parts[int(index_text)].label or "the result"
        if expr in self.variables:
            info = self.variables[expr]
            return info.label or info.phrase or "the result"
        if match := re.fullmatch(r"(\w+)\.(\w+)", expr):
            _, field = match.groups()
            return f"the {humanize_name(field)}"
        if "+" in expr:
            parts = [part.strip() for part in expr.split("+")]
            rendered: list[str] = []
            for part in parts:
                literal = parse_literal(part)
                if isinstance(literal, str) and part.startswith('"'):
                    rendered.append(f"'{literal}'")
                else:
                    rendered.append(self._describe_expression(part))
            return " followed by ".join(rendered)
        literal = parse_literal(expr)
        if isinstance(literal, str):
            return f"'{literal}'"
        return format_number(literal)

    def _describe_prints(self, expressions: list[str]) -> str:
        if len(expressions) == 1:
            return f"print {self._describe_expression(expressions[0])}"
        if self._same_parallel_results(expressions):
            return "print both results"
        described = [self._describe_expression(expression) for expression in expressions]
        if all(item.startswith("the ") for item in described):
            return f"print the {oxford_join([item[4:] for item in described])}"
        return f"print {oxford_join(described)}"

    def _same_parallel_results(self, expressions: list[str]) -> bool:
        matches = [re.fullmatch(r"(\w+)\[(\d+)\]", expr.strip()) for expr in expressions]
        if not all(matches):
            return False
        var_names = {match.group(1) for match in matches if match}
        indexes = sorted(int(match.group(2)) for match in matches if match)
        return len(var_names) == 1 and indexes == list(range(len(expressions)))

    def _describe_commits(self, commits: list[tuple[str, str]]) -> str:
        names = [name for name, _ in commits]
        if len(names) == 1:
            return f"commit the result as '{names[0]}'"
        rendered = [f"'{name}'" for name in names]
        return f"commit the results as {oxford_join(rendered)}"


def validate_intent(intent: str, branescript: str) -> list[str]:
    errors: list[str] = []
    for raw_patient in STRING_ASSIGN_RE.findall(branescript):
        _, raw_value = raw_patient
        value = decode_brane_string(raw_value)
        if not value.startswith("{"):
            continue
        try:
            patient = json.loads(value)
        except json.JSONDecodeError:
            continue
        if {"patient_id", "age", "gender"} <= patient.keys():
            expected_values = [
                str(patient["patient_id"]),
                str(patient["age"]),
                str(patient["vital_signs"]["blood_pressure"]),
                str(patient["vital_signs"]["heart_rate"]),
                str(patient["vital_signs"]["temperature"]),
                str(patient["vital_signs"]["spo2"]),
                str(patient["vital_signs"]["weight_kg"]),
                str(patient["vital_signs"]["height_cm"]),
                str(patient["lab_results"]["total_cholesterol"]),
                str(patient["lab_results"]["glucose"]),
                str(patient["lab_results"]["hba1c"]),
            ]
            for expected in expected_values:
                if expected not in intent:
                    errors.append(f"Missing patient value {expected}")
            history = patient.get("medical_history", [])
            if history:
                for item in history:
                    if item not in intent:
                        errors.append(f"Missing medical history item {item}")
            elif "no known medical history" not in intent:
                errors.append("Missing no known medical history")
    for _, dataset_name in DATA_ASSIGN_RE.findall(branescript):
        if dataset_name not in intent:
            errors.append(f"Missing dataset name {dataset_name}")
    for _, raw_value in STRING_ASSIGN_RE.findall(branescript):
        value = decode_brane_string(raw_value)
        if value.startswith("{") or value.startswith("["):
            continue
        used_in_text_operation = any(
            func in branescript
            for func in [
                "count_words(",
                "count_sentences(",
                "compute_readability(",
                "compute_sentiment(",
                "extract_keywords(",
                "get_text_stats(",
                "detect_pii(",
                "mask_value(",
            ]
        )
        if used_in_text_operation and value not in intent:
            errors.append(f"Missing text content {value}")
    commit_names = [name for name, _ in COMMIT_RE.findall(branescript)]
    if commit_names:
        if "commit" not in intent:
            errors.append("Missing commit wording")
        for name in commit_names:
            if name not in intent:
                errors.append(f"Missing commit name {name}")
    if PRINT_RE.search(branescript) and "print" not in intent:
        errors.append("Missing print wording")
    return errors


def rewrite_file(path: Path) -> tuple[int, int]:
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    original_count = len(entries)
    for entry in entries:
        intent = IntentGenerator(entry["branescript"]).generate()
        errors = validate_intent(intent, entry["branescript"])
        if errors:
            raise ValueError(f"{path}:{entry.get('id', '<no-id>')} -> {'; '.join(errors)}")
        entry["intent"] = intent
    path.write_text("\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n")
    return original_count, len(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "data/examples/packages.jsonl",
            "data/examples/training_500.jsonl",
        ],
    )
    args = parser.parse_args()
    for raw_path in args.paths:
        path = Path(raw_path)
        before, after = rewrite_file(path)
        print(f"{path}: preserved {after} entries")
        if before != after:
            raise ValueError(f"{path}: entry count changed from {before} to {after}")


if __name__ == "__main__":
    main()
