#!/usr/bin/env python3
"""Rule-based paraphrase generator — no GPU needed."""
import json, re, os

INPUT_FILES = [
    "/home/manu/Documents/Thesis topic/NLP-Brane-Translator/data/training/train.jsonl",
    "/home/manu/Documents/Thesis topic/NLP-Brane-Translator/data/training/val.jsonl",
]
OUTPUT_FILE = "/home/manu/Documents/Thesis topic/NLP-Brane-Translator/data/training/paraphrases.jsonl"

# ---------------------------------------------------------------------------
# Utility substitutions
# ---------------------------------------------------------------------------

def sub_print(s, v):
    opts = ["and output", "and display", "then report"]
    return re.sub(r"\band print\b", opts[v % 3], s)

def sub_commit(s, v):
    opts = ["and store the result as", "and save the output as", "and write the result as"]
    return re.sub(r"and commit the results? as", opts[v % 3], s)

def sub_in_ds(s, v):
    opts = [r"from the \1 dataset", r"within the \1 dataset", r"across the \1 dataset"]
    return re.sub(r"\bin the (\w+) dataset\b", opts[v % 3], s, count=1)

# ---------------------------------------------------------------------------
# Pattern-specific paraphrasers
# ---------------------------------------------------------------------------

def pat_patient(prefix_re, op_syns, intent, v):
    """Handle intents that operate on a single patient with vitals."""
    m = re.match(
        prefix_re +
        r"a (\d+)-year-old (male|female|non-binary)? *patient \((\w+)\) with (.+?)(, and print |, and output |, and display |, then report )(.+)",
        intent, re.DOTALL)
    if not m:
        return None
    offset = m.lastindex - 5  # groups before our standard ones
    age = m.group(offset + 1)
    gender = (m.group(offset + 2) or "").strip()
    pid = m.group(offset + 3)
    vitals = m.group(offset + 4)
    outputs = m.group(offset + 6)
    op = op_syns[v % len(op_syns)]
    if v == 0:
        return f"{op} patient {pid}, a {age}-year-old {gender} with {vitals}, and output {outputs}"
    elif v == 1:
        return f"For {pid}, aged {age}, {gender}, presenting with {vitals}, {op.lower()} and display {outputs}"
    else:
        return f"Given a {age}-year-old {gender} patient ({pid}) with {vitals}, {op.lower()} and report {outputs}"


def para_triage_single(intent, v):
    return pat_patient(r"Triage ", ["Perform triage on", "Assess the triage level for", "Carry out triage for"], intent, v)

def para_vitals(intent, v):
    return pat_patient(r"Check the vital signs for ", ["Evaluate the vital signs of", "Assess the vital parameters for", "Review the vitals for"], intent, v)

def para_mortality(intent, v):
    return pat_patient(r"Compute mortality risk for ", ["Estimate the mortality risk for", "Calculate the mortality risk for", "Determine the mortality risk for"], intent, v)

def para_diabetes_single(intent, v):
    return pat_patient(r"Assess diabetes risk for ", ["Evaluate the diabetes risk for", "Determine the diabetes risk level for", "Calculate the diabetes risk for"], intent, v)

def para_heart_single(intent, v):
    return pat_patient(r"Analyze heart disease risk for ", ["Evaluate heart disease risk for", "Assess cardiovascular risk for", "Perform heart disease risk analysis for"], intent, v)

def para_health_report(intent, v):
    return pat_patient(r"Generate a full health report for ", ["Produce a comprehensive health report for", "Create a full health report for", "Build a complete health report for"], intent, v)

def para_patient_summary(intent, v):
    return pat_patient(r"Get a patient summary for ", ["Retrieve the patient summary for", "Generate a summary for", "Obtain a patient summary for"], intent, v)

def para_readmission(intent, v):
    return pat_patient(r"Predict 30-day hospital readmission risk for ", ["Estimate the 30-day hospital readmission risk for", "Calculate 30-day readmission probability for", "Forecast the 30-day readmission risk for"], intent, v)

def para_validate(intent, v):
    return pat_patient(r"Validate the data for ", ["Verify the clinical data for", "Check the data validity for", "Run data validation for"], intent, v)

def para_heart_named(intent, v):
    m = re.match(r"Analyze heart disease risk for patient (\w+), a (\d+)-year-old (male|female)(.*)", intent)
    if not m:
        return None
    pid, age, gender, rest = m.groups()
    rest_p = sub_print(rest, v)
    ops = ["Evaluate heart disease risk for patient", "Assess cardiovascular risk for", "Perform heart disease risk analysis for patient"]
    if v == 0:
        return f"{ops[0]} {pid}, a {age}-year-old {gender}{rest_p}"
    elif v == 1:
        return f"{ops[1]} {pid}, aged {age}, {gender}{rest_p}"
    else:
        return f"{ops[2]} {pid} ({age}-year-old {gender}){rest_p}"

def para_summary_stats(intent, v):
    m = re.match(r"Compute summary statistics for (?:the )?(\w+) column (?:in|on|from|of) the (\w+) dataset(.*)", intent)
    if not m:
        return None
    col, ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate descriptive statistics for the {col} column in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Generate summary statistics on the {col} variable from the {ds} dataset{rest_p}"
    else:
        return f"Summarise the {col} column in the {ds} dataset{rest_p}"

def para_cardio_all(intent, v):
    m = re.match(r"Run cardiovascular analysis on all patients in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    ops = ["Perform cardiovascular analysis on all patients in", "Apply cardiovascular analysis to every patient in", "Conduct a cardiovascular analysis across all patients in"]
    return f"{ops[v % 3]} the {ds} dataset{rest_p}"

def para_classify_epidemic(intent, v):
    m = re.match(r"Classify the epidemic stage for the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    ops = ["Determine the epidemic stage for", "Identify the epidemic classification for", "Categorise the epidemic stage using"]
    return f"{ops[v % 3]} the {ds} dataset{rest_p}"

def para_estimate_reproduction(intent, v):
    m = re.match(r"Estimate the reproduction number for the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    ops = ["Calculate the reproduction number for", "Compute the epidemic reproduction number from", "Determine the reproduction number using"]
    return f"{ops[v % 3]} the {ds} dataset{rest_p}"

def para_detect_pii(intent, v):
    m = re.match(r"Detect personally identifiable information in the text '(.+?)'(.*)", intent, re.DOTALL)
    if not m:
        return None
    text, rest = m.groups()
    rest_p = sub_print(rest, v)
    ops = ["Identify personally identifiable information in the text", "Scan the text for personally identifiable information:", "Find all PII present in the text"]
    if v == 0:
        return f"Identify personally identifiable information in the text '{text}'{rest_p}"
    elif v == 1:
        return f"Scan the text '{text}' for personally identifiable information{rest_p}"
    else:
        return f"Find all PII present in the text '{text}'{rest_p}"

def para_filter_records(intent, v):
    m = re.match(r"Filter the records in the (\w+) dataset where (.+?), and commit the result as '(.+?)'(.*)", intent)
    if not m:
        return None
    ds, cond, name, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Extract records from the {ds} dataset where {cond} and store the result as '{name}'{rest_p}"
    elif v == 1:
        return f"Select all rows in the {ds} dataset satisfying {cond} and save the output as '{name}'{rest_p}"
    else:
        return f"Retrieve records from the {ds} dataset meeting the condition {cond} and write the result as '{name}'{rest_p}"

def para_filter_all_patients(intent, v):
    m = re.match(r"Filter all patients in the (\w+) dataset to only (.+?), and commit the result as '(.+?)'(.*)", intent)
    if not m:
        return None
    ds, crit, name, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Select only {crit} from the {ds} dataset and save as '{name}'{rest_p}"
    elif v == 1:
        return f"Restrict the {ds} dataset to {crit} patients and store the result as '{name}'{rest_p}"
    else:
        return f"Keep only the {crit} in the {ds} dataset and write the filtered data as '{name}'{rest_p}"

def para_detect_outliers(intent, v):
    m = re.match(r"Detect outliers in the (\w+) column of the (\w+) dataset using the (\w+) method(.*)", intent)
    if not m:
        return None
    col, ds, method, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Identify outliers in the {col} column of the {ds} dataset using the {method} method{rest_p}"
    elif v == 1:
        return f"Find anomalous values in the {col} column of the {ds} dataset via the {method} approach{rest_p}"
    else:
        return f"Apply the {method} method to detect outliers in the {col} column of the {ds} dataset{rest_p}"

def para_normalize(intent, v):
    m = re.match(r"Normalize the (\w+) column in the (\w+) dataset using (\w+) normalization, and commit the result as '(.+?)'(.*)", intent)
    if not m:
        return None
    col, ds, method, name, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Apply {method} normalisation to the {col} column in the {ds} dataset and save as '{name}'{rest_p}"
    elif v == 1:
        return f"Scale the {col} column in the {ds} dataset using {method} normalisation and store as '{name}'{rest_p}"
    else:
        return f"Standardise the {col} column of the {ds} dataset with {method} normalisation and write the result as '{name}'{rest_p}"

def para_correlation(intent, v):
    m = re.match(r"Compute the correlation between (\w+) and (\w+) in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    c1, c2, ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the correlation between {c1} and {c2} in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Determine the statistical relationship between {c1} and {c2} in the {ds} dataset{rest_p}"
    else:
        return f"Find the Pearson correlation of {c1} and {c2} from the {ds} dataset{rest_p}"

def para_mean_grouped(intent, v):
    m = re.match(r"Compute the mean (\w+) grouped by (\w+) in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    col, grp, ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the average {col} grouped by {grp} in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Find the mean {col} for each {grp} in the {ds} dataset{rest_p}"
    else:
        return f"Determine the mean {col} broken down by {grp} in the {ds} dataset{rest_p}"

def para_sum_grouped(intent, v):
    m = re.match(r"Compute the sum (\w+) grouped by (\w+) in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    col, grp, ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the total {col} grouped by {grp} in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Find the sum of {col} broken down by {grp} in the {ds} dataset{rest_p}"
    else:
        return f"Determine the aggregate {col} for each {grp} in the {ds} dataset{rest_p}"

def para_max_grouped(intent, v):
    m = re.match(r"Compute the max (\w+) grouped by (\w+) in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    col, grp, ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Find the maximum {col} grouped by {grp} in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Determine the highest {col} for each {grp} in the {ds} dataset{rest_p}"
    else:
        return f"Calculate the max {col} broken down by {grp} in the {ds} dataset{rest_p}"

def para_max_new_cases(intent, v):
    m = re.match(r"Compute the max new cases grouped by (\w+) in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    grp, ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Find the maximum number of new cases grouped by {grp} in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Determine the peak new cases per {grp} in the {ds} dataset{rest_p}"
    else:
        return f"Calculate the highest new case count for each {grp} in the {ds} dataset{rest_p}"

def para_sort_rank(intent, v):
    m = re.match(r"Sort and rank the records in the (\w+) dataset by (\w+) in (ascending|descending) order, and commit the result as '(.+?)'(.*)", intent)
    if not m:
        return None
    ds, col, order, name, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Order and rank the records in the {ds} dataset by {col} ({order}) and save the result as '{name}'{rest_p}"
    elif v == 1:
        return f"Rank all records in the {ds} dataset by {col} in {order} order and store as '{name}'{rest_p}"
    else:
        return f"Sort the {ds} dataset by {col} in {order} order and write the ranked records as '{name}'{rest_p}"

def para_count_records_by(intent, v):
    m = re.match(r"Count the records in the (\w+) dataset by (\w+)(.*)", intent)
    if not m:
        return None
    ds, col, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Tally the records in the {ds} dataset grouped by {col}{rest_p}"
    elif v == 1:
        return f"Count entries in the {ds} dataset broken down by {col}{rest_p}"
    else:
        return f"Group and count the records in the {ds} dataset by {col}{rest_p}"

def para_count_words(intent, v):
    m = re.match(r"Count the number of words in the text '(.+?)'(.*)", intent, re.DOTALL)
    if not m:
        return None
    text, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the word count of the text '{text}'{rest_p}"
    elif v == 1:
        return f"Tally the number of words in the text '{text}'{rest_p}"
    else:
        return f"Find how many words appear in the text '{text}'{rest_p}"

def para_count_sentences(intent, v):
    m = re.match(r"Count the number of sentences in the text '(.+?)'(.*)", intent, re.DOTALL)
    if not m:
        return None
    text, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Determine the number of sentences in the text '{text}'{rest_p}"
    elif v == 1:
        return f"Tally the sentence count in the text '{text}'{rest_p}"
    else:
        return f"Find how many sentences appear in the text '{text}'{rest_p}"

def para_sentiment(intent, v):
    m = re.match(r"Compute the sentiment of the text '(.+?)'(.*)", intent, re.DOTALL)
    if not m:
        return None
    text, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Analyse the sentiment of the text '{text}'{rest_p}"
    elif v == 1:
        return f"Determine the sentiment score for the text '{text}'{rest_p}"
    else:
        return f"Calculate the emotional polarity of the text '{text}'{rest_p}"

def para_readability(intent, v):
    m = re.match(r"Compute the readability score of the text '(.+?)'(.*)", intent, re.DOTALL)
    if not m:
        return None
    text, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the readability score of the text '{text}'{rest_p}"
    elif v == 1:
        return f"Determine how readable the text '{text}' is{rest_p}"
    else:
        return f"Measure the readability of the text '{text}'{rest_p}"

def para_mask_value(intent, v):
    m = re.match(r"Mask the value '(.+?)' using the (\w+) strategy(.*)", intent)
    if not m:
        return None
    val, strategy, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Apply the {strategy} masking strategy to the value '{val}'{rest_p}"
    elif v == 1:
        return f"Redact the value '{val}' using the {strategy} approach{rest_p}"
    else:
        return f"Hide the value '{val}' by applying the {strategy} masking strategy{rest_p}"

def para_mask_named_fields(intent, v):
    m = re.match(r"Mask the (.+?) (?:field|fields) in the record with (.+?), and print (.+)", intent, re.DOTALL)
    if not m:
        return None
    fields, record_desc, outputs = m.groups()
    if v == 0:
        return f"Anonymise the {fields} fields in the record with {record_desc}, and output {outputs}"
    elif v == 1:
        return f"Redact the {fields} fields from the record containing {record_desc}, and display {outputs}"
    else:
        return f"Apply masking to the {fields} fields of the record described by {record_desc}, and report {outputs}"

def para_mask_field_dataset(intent, v):
    m = re.match(r"Mask the (.+?) field(?:s)? in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    field, ds, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Anonymise the {field} field in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Redact the {field} field from the {ds} dataset{rest_p}"
    else:
        return f"Apply masking to the {field} field in the {ds} dataset{rest_p}"

def para_incidence_rate(intent, v):
    m = re.match(r"Compute the incidence rate in the (\w+) dataset per (\d+) people(.*)", intent)
    if not m:
        return None
    ds, pop, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the incidence rate per {pop} people in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Determine the incidence rate per {pop} population using the {ds} dataset{rest_p}"
    else:
        return f"Find the incidence rate for every {pop} people from the {ds} dataset{rest_p}"

def para_attack_rate(intent, v):
    m = re.match(r"Compute the attack rate for (\d+) confirmed cases out of (\d+) exposed people(.*)", intent)
    if not m:
        return None
    cases, pop, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the attack rate given {cases} confirmed cases among {pop} exposed individuals{rest_p}"
    elif v == 1:
        return f"Determine the attack rate for {cases} cases out of {pop} exposed people{rest_p}"
    else:
        return f"Find the attack rate when {cases} cases occur in a population of {pop} exposed persons{rest_p}"

def para_detect_outbreaks(intent, v):
    m = re.match(r"Detect outbreaks in the (\w+) dataset using a threshold of ([\d.]+)(.*)", intent)
    if not m:
        return None
    ds, thresh, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Identify outbreaks in the {ds} dataset with a threshold of {thresh}{rest_p}"
    elif v == 1:
        return f"Find outbreak events in the {ds} dataset using threshold {thresh}{rest_p}"
    else:
        return f"Scan the {ds} dataset for outbreaks with an alert threshold of {thresh}{rest_p}"

def para_bmi(intent, v):
    m = re.match(r"Compute the BMI for a patient weighing ([\d.]+)kg and measuring ([\d.]+)cm(.*)", intent)
    if not m:
        return None
    weight, height, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the BMI for a patient with a weight of {weight}kg and height of {height}cm{rest_p}"
    elif v == 1:
        return f"Determine the BMI of a patient weighing {weight}kg and standing {height}cm tall{rest_p}"
    else:
        return f"Find the body mass index for a patient whose weight is {weight}kg and height is {height}cm{rest_p}"

def para_prevalence(intent, v):
    m = re.match(r"Compute the prevalence of risk factors in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the prevalence of risk factors in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Determine the risk factor prevalence within the {ds} dataset{rest_p}"
    else:
        return f"Find the proportion of risk factors present in the {ds} dataset{rest_p}"

def para_epidemic_status(intent, v):
    m = re.match(r"Get the epidemic status (?:for|from) the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Retrieve the epidemic status for the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Obtain the epidemic status information from the {ds} dataset{rest_p}"
    else:
        return f"Fetch the current epidemic status using the {ds} dataset{rest_p}"

def para_epidemic_report(intent, v):
    m = re.match(r"Generate a full epidemic surveillance report for the (\w+) dataset using a population of (\d+)(.*)", intent)
    if not m:
        return None
    ds, pop, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Create a comprehensive epidemic surveillance report for the {ds} dataset with a population of {pop}{rest_p}"
    elif v == 1:
        return f"Produce a full epidemic surveillance report using the {ds} dataset and a population size of {pop}{rest_p}"
    else:
        return f"Build an epidemic surveillance report from the {ds} dataset for a population of {pop}{rest_p}"

def para_gc_content_multi(intent, v):
    m = re.match(r"Compute GC content for the sequences (.+?) and print (.+)", intent, re.DOTALL)
    if not m:
        return None
    seqs, outputs = m.groups()
    if v == 0:
        return f"Calculate the GC content for the sequences {seqs} and output {outputs}"
    elif v == 1:
        return f"Determine the GC content of each sequence in {seqs} and display {outputs}"
    else:
        return f"Find the GC content for each of the sequences {seqs} and report {outputs}"

def para_gc_content(intent, v):
    m = re.match(r"Compute GC content for (?:the )?sequence (\w+)(.*)", intent)
    if not m:
        return None
    seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the GC content of sequence {seq}{rest_p}"
    elif v == 1:
        return f"Determine the GC content for sequence {seq}{rest_p}"
    else:
        return f"Find the proportion of G and C bases in sequence {seq}{rest_p}"

def para_complement(intent, v):
    m = re.match(r"Get the (?:Watson-Crick )?complement of (?:the )?sequence (\w+)(.*)", intent)
    if not m:
        return None
    seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Compute the complementary strand for sequence {seq}{rest_p}"
    elif v == 1:
        return f"Determine the Watson-Crick complement of sequence {seq}{rest_p}"
    else:
        return f"Find the nucleotide complement of the sequence {seq}{rest_p}"

def para_reverse_complement(intent, v):
    m = re.match(r"Get the reverse complement of (?:the )?sequence (\w+)(.*)", intent)
    if not m:
        return None
    seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Compute the reverse complement of sequence {seq}{rest_p}"
    elif v == 1:
        return f"Determine the reverse complementary strand for sequence {seq}{rest_p}"
    else:
        return f"Find the reverse-complement sequence for {seq}{rest_p}"

def para_count_motifs(intent, v):
    m = re.match(r"Count (\w+) motifs? in (?:the )?sequence (\w+)(.*)", intent)
    if not m:
        return None
    motif, seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Find the number of {motif} motifs in sequence {seq}{rest_p}"
    elif v == 1:
        return f"Count occurrences of the {motif} motif in sequence {seq}{rest_p}"
    else:
        return f"Determine how many times {motif} appears in sequence {seq}{rest_p}"

def para_count_motif_how(intent, v):
    m = re.match(r"Count how many times the motif (\w+) appears in the sequence (\w+)(.*)", intent)
    if not m:
        return None
    motif, seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Find the number of occurrences of motif {motif} in sequence {seq}{rest_p}"
    elif v == 1:
        return f"Count how often {motif} occurs in sequence {seq}{rest_p}"
    else:
        return f"Determine the frequency of motif {motif} in sequence {seq}{rest_p}"

def para_analyze_sequence(intent, v):
    m = re.match(r"Analyze (?:the )?sequence (\w+)(.*)", intent)
    if not m:
        return None
    seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Perform sequence analysis on {seq}{rest_p}"
    elif v == 1:
        return f"Examine the sequence {seq}{rest_p}"
    else:
        return f"Run an analysis of the genomic sequence {seq}{rest_p}"

def para_translate_seq(intent, v):
    m = re.match(r"Translate (?:the )?sequence (\w+)(.*)", intent)
    if not m:
        return None
    seq, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Perform translation of sequence {seq}{rest_p}"
    elif v == 1:
        return f"Convert the sequence {seq} to its protein product{rest_p}"
    else:
        return f"Translate the nucleotide sequence {seq} into an amino acid chain{rest_p}"

def para_hamming(intent, v):
    m = re.match(r"Compute (?:the )?Hamming distance between (\w+) and (\w+)(.*)", intent)
    if not m:
        return None
    s1, s2, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the Hamming distance between {s1} and {s2}{rest_p}"
    elif v == 1:
        return f"Determine the edit distance (Hamming) between {s1} and {s2}{rest_p}"
    else:
        return f"Find the Hamming distance separating {s1} and {s2}{rest_p}"

def para_align_seqs(intent, v):
    m = re.match(r"Align (?:the )?sequences? (\w+) and (\w+)(.*)", intent, re.DOTALL)
    if not m:
        return None
    s1, s2, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Perform sequence alignment between {s1} and {s2}{rest_p}"
    elif v == 1:
        return f"Compare sequences {s1} and {s2} through alignment{rest_p}"
    else:
        return f"Run an alignment of {s1} against {s2}{rest_p}"

def para_triage_all(intent, v):
    m = re.match(r"Triage all patients in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Perform triage on all patients in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Run batch triage for every patient in the {ds} dataset{rest_p}"
    else:
        return f"Apply triage assessment to all patients in the {ds} dataset{rest_p}"

def para_diabetes_all(intent, v):
    m = re.match(r"Assess diabetes risk for all patients in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Evaluate the diabetes risk for all patients in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Calculate diabetes risk scores for every patient in the {ds} dataset{rest_p}"
    else:
        return f"Run a batch diabetes risk assessment on all patients in the {ds} dataset{rest_p}"

def para_text_stats(intent, v):
    m = re.match(r"Get text statistics (?:for|of) the text '(.+?)'(.*)", intent, re.DOTALL)
    if not m:
        return None
    text, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Retrieve text statistics for the text '{text}'{rest_p}"
    elif v == 1:
        return f"Calculate text metrics for the passage '{text}'{rest_p}"
    else:
        return f"Obtain statistical measures for the text '{text}'{rest_p}"

def para_extract_top(intent, v):
    m = re.match(r"Extract the top (.+?) from (.+?)(,| and )(.*)", intent, re.DOTALL)
    if not m:
        return None
    what, source, sep, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Retrieve the top {what} from {source} and {rest_p}"
    elif v == 1:
        return f"Find the highest-ranked {what} in {source} and {rest_p}"
    else:
        return f"Get the leading {what} from {source} and {rest_p}"

def para_analyze_texts_ds(intent, v):
    m = re.match(r"Analyze the texts in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Perform text analysis on the entries in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Process and analyse all texts in the {ds} dataset{rest_p}"
    else:
        return f"Run textual analysis across the {ds} dataset{rest_p}"

def para_parallel_tasks(intent, v):
    m = re.match(r"Run these in parallel: (.+)", intent, re.DOTALL)
    if not m:
        return None
    body = m.group(1)
    if v == 0:
        return f"Execute the following tasks concurrently: {body}"
    elif v == 1:
        return f"Perform in parallel: {body}"
    else:
        return f"Simultaneously carry out the following: {body}"

def para_parallel_site(intent, v):
    m = re.match(r"Run (.+?) in parallel on site '(.+?)' and site '(.+?)'(.+)", intent, re.DOTALL)
    if not m:
        return None
    op, s1, s2, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Execute {op} concurrently on site '{s1}' and site '{s2}'{rest_p}"
    elif v == 1:
        return f"Perform {op} simultaneously at site '{s1}' and '{s2}'{rest_p}"
    else:
        return f"Distribute {op} across site '{s1}' and site '{s2}'{rest_p}"

def para_tag_computation(intent, v):
    m = re.match(r"Tag the computation with metadata tag '(.+?)'(.*)", intent, re.DOTALL)
    if not m:
        return None
    tag, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Label the computation with the metadata tag '{tag}'{rest_p}"
    elif v == 1:
        return f"Attach the metadata tag '{tag}' to this computation{rest_p}"
    else:
        return f"Mark the computation using the metadata tag '{tag}'{rest_p}"

def para_load_dataset(intent, v):
    m = re.match(r"Load the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(sub_commit(rest, v), v)
    if v == 0:
        return f"Import the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Read in the {ds} dataset{rest_p}"
    else:
        return f"Fetch and load the {ds} dataset{rest_p}"

def para_cohort_stats(intent, v):
    m = re.match(r"Compute cohort statistics for the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate cohort statistics for the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Generate cohort-level statistics from the {ds} dataset{rest_p}"
    else:
        return f"Determine aggregate cohort statistics for the {ds} dataset{rest_p}"

def para_risk_distribution(intent, v):
    m = re.match(r"Compute the risk distribution from the cardiovascular analysis of all patients in the (\w+) dataset(.*)", intent)
    if not m:
        return None
    ds, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the risk distribution from the cardiovascular analysis across all patients in the {ds} dataset{rest_p}"
    elif v == 1:
        return f"Determine the cardiovascular risk distribution for all patients in the {ds} dataset{rest_p}"
    else:
        return f"Find the risk distribution resulting from cardiovascular analysis of all patients in the {ds} dataset{rest_p}"

def para_define_fn(intent, v):
    m = re.match(r"Define (a |two |three )?(.+?)(function|functions)(.*)", intent, re.DOTALL)
    if not m:
        return None
    qty, what, fn, rest = m.groups()
    rest_p = sub_print(rest, v)
    qty = qty or ""
    if v == 0:
        return f"Create {qty}{what}{fn}{rest_p}"
    elif v == 1:
        return f"Write {qty}{what}{fn}{rest_p}"
    else:
        return f"Implement {qty}{what}{fn}{rest_p}"

def para_declare(intent, v):
    m = re.match(r"Declare (.+?), and print (.+)", intent, re.DOTALL)
    if not m:
        return None
    decl, outputs = m.groups()
    if v == 0:
        return f"Define {decl}, and output {outputs}"
    elif v == 1:
        return f"Create {decl} and display {outputs}"
    else:
        return f"Set up {decl}, then report {outputs}"

def para_compute_gc_for(intent, v):
    # "Compute GC content for SEQUENCE and ..."
    m = re.match(r"Compute GC content for (\w+)(.*)", intent)
    if not m:
        return None
    seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the GC content for {seq}{rest_p}"
    elif v == 1:
        return f"Determine the GC content of {seq}{rest_p}"
    else:
        return f"Find the GC fraction of the sequence {seq}{rest_p}"

def para_compare_hamming(intent, v):
    m = re.match(r"Compare sequences (\w+) and (\w+) with Hamming distance(.*)", intent)
    if not m:
        return None
    s1, s2, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Calculate the Hamming distance between {s1} and {s2}{rest_p}"
    elif v == 1:
        return f"Measure the Hamming distance separating sequences {s1} and {s2}{rest_p}"
    else:
        return f"Find the edit distance (Hamming) between {s1} and {s2}{rest_p}"

def para_align_multi(intent, v):
    m = re.match(r"Align the (?:reference )?(\w+) against (.+?) and print (.+)", intent, re.DOTALL)
    if not m:
        return None
    ref, targets, outputs = m.groups()
    if v == 0:
        return f"Perform alignment of {ref} against {targets} and output {outputs}"
    elif v == 1:
        return f"Compare {ref} with {targets} using sequence alignment and display {outputs}"
    else:
        return f"Run pairwise alignment between {ref} and {targets} and report {outputs}"

def para_reverse_sequence(intent, v):
    m = re.match(r"Compute the reverse (?:complement )?(?:of )?(?:the )?sequence (\w+)(.*)", intent)
    if not m:
        return None
    seq, rest = m.groups()
    rest_p = sub_print(rest, v)
    if v == 0:
        return f"Find the reverse complement of the sequence {seq}{rest_p}"
    elif v == 1:
        return f"Determine the reverse of sequence {seq}{rest_p}"
    else:
        return f"Calculate the reverse complement for the sequence {seq}{rest_p}"

# ---------------------------------------------------------------------------
# Ordered list of all pattern functions
# ---------------------------------------------------------------------------
PATTERNS = [
    para_triage_single, para_vitals, para_mortality, para_diabetes_single,
    para_heart_single, para_health_report, para_patient_summary,
    para_readmission, para_validate, para_heart_named,
    para_parallel_site, para_tag_computation,
    para_summary_stats, para_filter_records, para_filter_all_patients,
    para_cardio_all, para_classify_epidemic, para_estimate_reproduction,
    para_detect_pii, para_detect_outliers, para_normalize,
    para_correlation, para_mean_grouped, para_sum_grouped,
    para_max_new_cases, para_max_grouped,
    para_sort_rank, para_count_records_by,
    para_count_words, para_count_sentences,
    para_sentiment, para_readability,
    para_mask_value, para_mask_named_fields, para_mask_field_dataset,
    para_incidence_rate, para_attack_rate, para_detect_outbreaks,
    para_bmi, para_prevalence, para_epidemic_status, para_epidemic_report,
    para_gc_content_multi, para_gc_content, para_compute_gc_for,
    para_complement, para_reverse_complement, para_reverse_sequence,
    para_count_motif_how, para_count_motifs,
    para_align_multi, para_align_seqs,
    para_compare_hamming, para_hamming,
    para_analyze_sequence, para_translate_seq,
    para_triage_all, para_diabetes_all,
    para_text_stats, para_extract_top, para_analyze_texts_ds,
    para_parallel_tasks, para_load_dataset,
    para_cohort_stats, para_risk_distribution,
    para_define_fn, para_declare,
]

def generic_para(intent, v):
    s = intent
    verb_maps = [
        (r"\bCompute\b", ["Calculate", "Determine", "Find"]),
        (r"\bAnalyze\b", ["Assess", "Evaluate", "Examine"]),
        (r"\bGenerate\b", ["Produce", "Create", "Build"]),
        (r"\bGet\b", ["Retrieve", "Obtain", "Fetch"]),
        (r"\bDetect\b", ["Identify", "Find", "Locate"]),
        (r"\bRun\b", ["Execute", "Perform", "Carry out"]),
    ]
    print_map = ["and output", "and display", "then report"]
    commit_map = ["and store the result as", "and save the output as", "and write the result as"]
    for pattern, syns in verb_maps:
        if re.search(pattern, s):
            s = re.sub(pattern, syns[v % len(syns)], s, count=1)
            break
    s = s.replace("and print", print_map[v % 3])
    s = re.sub(r"and commit the results? as", commit_map[v % 3], s)
    if s == intent:
        prefixes = ["Please ", "Now ", "Next, "]
        s = prefixes[v % 3] + s[0].lower() + s[1:]
    return s

def make_paraphrases(intent):
    results = []
    for v in range(3):
        result = None
        for fn in PATTERNS:
            try:
                result = fn(intent, v)
            except Exception:
                result = None
            if result is not None:
                break
        if result is None or result == intent:
            result = generic_para(intent, v)
        if result == intent:
            result = ("Please " + intent[0].lower() + intent[1:]) if v == 0 else \
                     ("Now " + intent[0].lower() + intent[1:]) if v == 1 else \
                     ("Next, " + intent[0].lower() + intent[1:])
        results.append(result)
    return results

def load_done_ids():
    done = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add(obj["original_id"])
                except Exception:
                    pass
    return done

def main():
    examples = []
    for fname in INPUT_FILES:
        with open(fname) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                examples.append({"id": obj["id"], "intent": obj["intent"]})
    print(f"Loaded {len(examples)} examples.")

    done_ids = load_done_ids()
    print(f"Already done: {len(done_ids)} examples.")

    to_process = [e for e in examples if e["id"] not in done_ids]
    print(f"To process: {len(to_process)} examples.")

    with open(OUTPUT_FILE, "a") as out:
        for i, ex in enumerate(to_process):
            paras = make_paraphrases(ex["intent"])
            for j, para in enumerate(paras, 1):
                record = {
                    "id": f"{ex['id']}_para_{j}",
                    "original_id": ex["id"],
                    "source_file": "paraphrases",
                    "intent": para,
                    "branescript": "",
                }
                out.write(json.dumps(record) + "\n")
            if (i + 1) % 50 == 0:
                out.flush()
                print(f"  Processed {i + 1} / {len(to_process)}")
        out.flush()

    with open(OUTPUT_FILE) as f:
        total = sum(1 for l in f if l.strip())
    print(f"Done. Total lines in output: {total}")

if __name__ == "__main__":
    main()
