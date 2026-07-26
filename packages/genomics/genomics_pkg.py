#!/usr/bin/env python3
"""
Genomics Package for NLP-Brane-Translator

Pure-computation genomics utilities operating on DNA sequences provided as
strings. All functions are deterministic: same input always produces the
same output. No external state, files, or timestamps are used.

Brane input convention:
  - String inputs: JSON-encoded via uppercase env var (e.g. SEQUENCE)
  - String outputs: printed as  output: "<json-string>"
  - Class outputs: printed as YAML two-element list ["ClassName", {fields}]

Available actions
-----------------
compute_gc_content        -- GC percentage of a DNA sequence
get_complement            -- Watson-Crick complement (A↔T, G↔C)
get_reverse_complement    -- reverse complement (5'→3')
count_motif               -- non-overlapping motif occurrences
analyze_sequence          -- full SequenceStats (gc, at, length, codons)
translate_dna             -- DNA → protein (standard genetic code)
compute_hamming_distance  -- mismatch count between same-length sequences
align_sequences           -- pairwise alignment stats (AlignmentResult)
"""

import json
import os
import sys
import yaml


# ---------------------------------------------------------------------------
# Genetic code
# ---------------------------------------------------------------------------

_CODON_TABLE: dict = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

_COMPLEMENT_TABLE = str.maketrans('ATGCatgcNn', 'TACGtacgNn')


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def _clean(seq: str) -> str:
    """Uppercase, strip whitespace."""
    return seq.upper().strip()


def _valid_bases(seq: str) -> list:
    return [c for c in seq if c in 'ATGCN']


def _gc_content(seq: str) -> float:
    bases = _valid_bases(seq)
    if not bases:
        return 0.0
    gc = sum(1 for c in bases if c in 'GC')
    return round(gc / len(bases) * 100, 2)


def _complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT_TABLE)


def _reverse_complement(seq: str) -> str:
    return _complement(seq)[::-1]


def _count_motif(seq: str, motif: str) -> int:
    seq = seq.upper()
    motif = motif.upper()
    if not motif:
        return 0
    count = 0
    start = 0
    while True:
        pos = seq.find(motif, start)
        if pos == -1:
            break
        count += 1
        start = pos + len(motif)
    return count


def _analyze_sequence(seq: str) -> dict:
    seq = _clean(seq)
    bases = _valid_bases(seq)
    length = len(bases)
    gc = sum(1 for c in bases if c in 'GC')
    at = sum(1 for c in bases if c in 'AT')
    gc_pct = round(gc / length * 100, 2) if length > 0 else 0.0
    at_pct = round(at / length * 100, 2) if length > 0 else 0.0
    gc_at_ratio = round(gc / at, 2) if at > 0 else 0.0
    has_start = seq.startswith('ATG')
    has_stop = 'TAA' in seq or 'TAG' in seq or 'TGA' in seq
    return {
        'length': length,
        'gc_content': gc_pct,
        'at_content': at_pct,
        'gc_at_ratio': gc_at_ratio,
        'has_start_codon': 'true' if has_start else 'false',
        'has_stop_codon': 'true' if has_stop else 'false',
    }


def _translate_dna(seq: str) -> str:
    seq = _clean(seq)
    protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        aa = _CODON_TABLE.get(codon, '?')
        if aa == '*':
            break
        protein.append(aa)
    return ''.join(protein)


def _hamming_distance(seq_a: str, seq_b: str) -> int:
    seq_a = _clean(seq_a)
    seq_b = _clean(seq_b)
    if len(seq_a) != len(seq_b):
        return -1
    return sum(a != b for a, b in zip(seq_a, seq_b))


def _align_sequences(seq_a: str, seq_b: str) -> dict:
    seq_a = _clean(seq_a)
    seq_b = _clean(seq_b)
    length = max(len(seq_a), len(seq_b))
    a_pad = seq_a.ljust(length, '-')
    b_pad = seq_b.ljust(length, '-')
    matches = sum(a == b and a != '-' for a, b in zip(a_pad, b_pad))
    mismatches = sum(a != b and a != '-' and b != '-' for a, b in zip(a_pad, b_pad))
    gaps = sum(a == '-' or b == '-' for a, b in zip(a_pad, b_pad))
    identity_pct = round(matches / length * 100, 1) if length > 0 else 0.0
    return {
        'query': seq_a,
        'subject': seq_b,
        'identity_pct': identity_pct,
        'length': length,
        'mismatches': mismatches,
        'gaps': gaps,
    }


# ---------------------------------------------------------------------------
# Brane I/O helpers
# ---------------------------------------------------------------------------

def _env_str(name: str) -> str:
    raw = os.environ.get(name.upper(), '""')
    try:
        v = json.loads(raw)
        return str(v)
    except (json.JSONDecodeError, ValueError):
        return raw


def _out_str(value: str) -> None:
    print(f'output: {json.dumps(str(value))}', flush=True)


def _out_class(class_name: str, fields: dict) -> None:
    print(yaml.dump({'output': [class_name, fields]}), end='', flush=True)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def action_compute_gc_content() -> None:
    seq = _env_str('SEQUENCE')
    result = _gc_content(_clean(seq))
    _out_str(str(result))


def action_get_complement() -> None:
    seq = _env_str('SEQUENCE')
    _out_str(_complement(seq))


def action_get_reverse_complement() -> None:
    seq = _env_str('SEQUENCE')
    _out_str(_reverse_complement(seq))


def action_count_motif() -> None:
    seq = _env_str('SEQUENCE')
    motif = _env_str('MOTIF')
    _out_str(str(_count_motif(seq, motif)))


def action_analyze_sequence() -> None:
    seq = _env_str('SEQUENCE')
    stats = _analyze_sequence(seq)
    _out_class('SequenceStats', stats)


def action_translate_dna() -> None:
    seq = _env_str('SEQUENCE')
    _out_str(_translate_dna(seq))


def action_compute_hamming_distance() -> None:
    seq_a = _env_str('SEQ_A')
    seq_b = _env_str('SEQ_B')
    _out_str(str(_hamming_distance(seq_a, seq_b)))


def action_align_sequences() -> None:
    seq_a = _env_str('SEQ_A')
    seq_b = _env_str('SEQ_B')
    result = _align_sequences(seq_a, seq_b)
    _out_class('AlignmentResult', result)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ACTIONS = {
    'compute_gc_content':       action_compute_gc_content,
    'get_complement':           action_get_complement,
    'get_reverse_complement':   action_get_reverse_complement,
    'count_motif':              action_count_motif,
    'analyze_sequence':         action_analyze_sequence,
    'translate_dna':            action_translate_dna,
    'compute_hamming_distance': action_compute_hamming_distance,
    'align_sequences':          action_align_sequences,
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
