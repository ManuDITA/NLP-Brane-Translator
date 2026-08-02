#!/usr/bin/env python3
"""
Text Analysis Package for NLP-Brane-Translator

Brane entrypoint: the action name is read from sys.argv[1] (set via
container.yml command.args). Input arguments arrive as uppercase env vars
whose values are JSON-serialised by branelet (e.g. TEXT, TOP_N).

All analysis uses pure Python stdlib — no third-party NLP libraries needed.

Function categories
-------------------
String-based (inline text via TEXT env var)
  count_words         -- total word count (integer)
  count_sentences     -- total sentence count (integer)
  extract_keywords    -- top-N keywords by TF weight (JSON string)
  compute_sentiment   -- returns SentimentResult with polarity, score, word count
  compute_readability -- returns ReadabilityScore with Flesch score and grade

Class-returning
  get_text_stats      -- returns TextStats instance

Data-based (Brane dataset: plain-text file; path via TEXT_FILE env var)
  analyze_text_file   -- writes full analysis JSON to /result/

IntermediateResult chaining
  generate_frequency_report -- reads analysis IR, writes CSV + HTML to /result/
"""
import csv
import json
import math
import os
import re
import sys
import yaml
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Stop-words (common English words excluded from keyword extraction)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'not', 'no', 'nor',
    'so', 'yet', 'both', 'either', 'neither', 'each', 'every', 'all',
    'this', 'that', 'these', 'those', 'it', 'its', 'i', 'me', 'my', 'we',
    'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'they', 'them',
    'their', 'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
    'as', 'if', 'then', 'than', 'also', 'just', 'more', 'most', 'very',
    'into', 'over', 'after', 'before', 'about', 'up', 'out', 'there',
}

# ---------------------------------------------------------------------------
# Sentiment lexicon (positive and negative words with weights)
# ---------------------------------------------------------------------------

_POS_WORDS = {
    'good': 1, 'great': 2, 'excellent': 2, 'amazing': 2, 'wonderful': 2,
    'fantastic': 2, 'outstanding': 2, 'superb': 2, 'brilliant': 2, 'best': 2,
    'love': 2, 'like': 1, 'enjoy': 1, 'happy': 1, 'glad': 1, 'pleased': 1,
    'positive': 1, 'benefit': 1, 'helpful': 1, 'useful': 1, 'effective': 1,
    'success': 1, 'successful': 1, 'improve': 1, 'better': 1, 'perfect': 2,
    'beautiful': 2, 'clean': 1, 'clear': 1, 'safe': 1, 'healthy': 1,
    'innovative': 1, 'creative': 1, 'efficient': 1, 'reliable': 1,
    'powerful': 1, 'strong': 1, 'smart': 1, 'clever': 1, 'well': 1,
    'correct': 1, 'right': 1, 'ideal': 1, 'optimal': 1, 'nice': 1,
}

_NEG_WORDS = {
    'bad': 1, 'poor': 1, 'terrible': 2, 'awful': 2, 'horrible': 2,
    'dreadful': 2, 'worst': 2, 'hate': 2, 'dislike': 1, 'disappointed': 1,
    'negative': 1, 'problem': 1, 'issue': 1, 'error': 1, 'fail': 1,
    'failure': 2, 'broken': 1, 'bug': 1, 'wrong': 1, 'incorrect': 1,
    'danger': 2, 'dangerous': 2, 'harmful': 2, 'toxic': 2, 'risk': 1,
    'difficult': 1, 'hard': 1, 'slow': 1, 'weak': 1, 'limited': 1,
    'worse': 1, 'missing': 1, 'lost': 1, 'corrupt': 2, 'invalid': 1,
    'ugly': 1, 'useless': 1, 'inefficient': 1, 'expensive': 1, 'waste': 1,
}

_NEGATORS = {'not', 'no', "n't", 'never', 'neither', 'nor', 'hardly', 'barely'}


# ---------------------------------------------------------------------------
# Core text analysis logic
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-zA-Z']+", text.lower())


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    parts = re.split(r'[.!?]+', text)
    return [s.strip() for s in parts if s.strip()]


def _count_syllables(word: str) -> int:
    """Approximate English syllable count using vowel-group heuristic."""
    word = word.lower().rstrip('e')
    count = len(re.findall(r'[aeiou]+', word))
    return max(1, count)


def _word_frequencies(tokens: List[str]) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in tokens:
        if len(t) > 1:
            freq[t] = freq.get(t, 0) + 1
    return freq


def _keyword_scores(tokens: List[str], top_n: int) -> List[Dict[str, Any]]:
    """Score keywords by TF after stop-word removal."""
    content = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    total = len(content) or 1
    freq = _word_frequencies(content)
    scored = [
        {'keyword': w, 'frequency': c, 'score': round(c / total, 4)}
        for w, c in freq.items()
    ]
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:max(1, top_n)]


def _sentiment(tokens: List[str]) -> Tuple[str, float]:
    """Simple lexicon-based sentiment with negation handling."""
    score = 0.0
    prev_negated = False
    for token in tokens:
        if token in _NEGATORS:
            prev_negated = True
            continue
        multiplier = -1 if prev_negated else 1
        prev_negated = False
        if token in _POS_WORDS:
            score += multiplier * _POS_WORDS[token]
        elif token in _NEG_WORDS:
            score -= multiplier * _NEG_WORDS[token]

    norm = max(1, len(tokens) / 10)
    normalised = max(-1.0, min(1.0, score / norm))
    polarity = 'positive' if normalised > 0.05 else ('negative' if normalised < -0.05 else 'neutral')
    return polarity, round(normalised, 4)


def _readability(text: str) -> Tuple[float, str]:
    """Flesch Reading Ease score + approximate US grade level label."""
    words = _tokenise(text)
    sentences = _split_sentences(text)
    n_words = len(words) or 1
    n_sentences = len(sentences) or 1
    n_syllables = sum(_count_syllables(w) for w in words)

    flesch = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
    flesch = round(max(0.0, min(100.0, flesch)), 2)

    if flesch >= 90:
        grade = '5th grade (very easy)'
    elif flesch >= 80:
        grade = '6th grade (easy)'
    elif flesch >= 70:
        grade = '7th grade (fairly easy)'
    elif flesch >= 60:
        grade = '8th–9th grade (standard)'
    elif flesch >= 50:
        grade = '10th–12th grade (fairly difficult)'
    elif flesch >= 30:
        grade = 'College level (difficult)'
    else:
        grade = 'Professional (very difficult)'

    return flesch, grade


def _full_analysis(text: str) -> Dict[str, Any]:
    """Run all analyses on `text` and return a consolidated dict."""
    tokens = _tokenise(text)
    sentences = _split_sentences(text)
    n_words = len(tokens)
    n_sentences = len(sentences)
    avg_word_len = round(sum(len(w) for w in tokens) / max(1, n_words), 2)
    flesch, grade = _readability(text)
    polarity, score = _sentiment(tokens)
    keywords = _keyword_scores(tokens, top_n=20)
    freq = _word_frequencies(tokens)

    return {
        'word_count': n_words,
        'sentence_count': n_sentences,
        'avg_word_length': avg_word_len,
        'readability_score': flesch,
        'readability_grade': grade,
        'sentiment': polarity,
        'sentiment_score': score,
        'top_keywords': keywords,
        'word_frequencies': dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:50]),
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
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


def _env_int(name: str, default: int = 10) -> int:
    raw = os.environ.get(name.upper(), str(default))
    try:
        return int(json.loads(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def _resolve_env_path(var_name: str) -> str:
    raw = os.environ.get(var_name, '')
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, str):
            return decoded
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _out_str(value: str) -> None:
    """Print a string output; json.dumps ensures it stays a YAML string."""
    print(f'output: {json.dumps(str(value))}', flush=True)


def _out_int(value: int) -> None:
    print(f'output: {int(value)}', flush=True)


def _out_class(class_name: str, fields: Dict[str, Any]) -> None:
    """Print a class instance as a YAML 2-element list ["ClassName", {…}].

    serde_yaml deserialises this as FullValue::Instance(name, field_map).
    """
    print(yaml.dump({'output': [class_name, fields]}), end='', flush=True)


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def action_count_words() -> None:
    text = _env_str('TEXT')
    _out_int(len(_tokenise(text)))


def action_count_sentences() -> None:
    text = _env_str('TEXT')
    _out_int(len(_split_sentences(text)))


def action_extract_keywords() -> None:
    text = _env_str('TEXT')
    top_n = _env_int('TOP_N', default=10)
    keywords = _keyword_scores(_tokenise(text), top_n)
    _out_str(json.dumps(keywords))


def action_compute_sentiment() -> None:
    text = _env_str('TEXT')
    tokens = _tokenise(text)
    polarity, score = _sentiment(tokens)
    _out_class('SentimentResult', {
        'polarity': polarity,
        'score': round(float(score), 4),
        'word_count': len(tokens),
    })


def action_compute_readability() -> None:
    text = _env_str('TEXT')
    flesch, grade = _readability(text)
    _out_class('ReadabilityScore', {
        'flesch_score': round(float(flesch), 2),
        'grade_level': str(grade),
    })


def action_get_text_stats() -> None:
    """Return a TextStats class instance.

    BraneScript usage:
        let s := get_text_stats("The quick brown fox...");
        println(s.word_count);
        if (s.sentiment == "positive") { println("Positive text!"); }
    """
    text = _env_str('TEXT')
    tokens = _tokenise(text)
    sentences = _split_sentences(text)
    n_words = len(tokens)
    avg_word_len = round(sum(len(w) for w in tokens) / max(1, n_words), 4)
    flesch, _ = _readability(text)
    polarity, _ = _sentiment(tokens)
    _out_class('TextStats', {
        'word_count': n_words,
        'sentence_count': len(sentences),
        'avg_word_length': avg_word_len,
        'readability_score': flesch,
        'sentiment': polarity,
    })


def action_analyze_text_file() -> None:
    """Data-based: read a plain-text file and write full analysis to /result/."""
    path = _resolve_env_path('TEXT_FILE')
    if not path:
        raise RuntimeError('TEXT_FILE not set — invoke via a Data reference')

    # Support single file or directory with .txt files
    if os.path.isdir(path):
        texts: Dict[str, str] = {}
        for fname in os.listdir(path):
            if fname.endswith('.txt'):
                fpath = os.path.join(path, fname)
                with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                    texts[fname] = f.read()
    else:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            texts = {os.path.basename(path): f.read()}

    results = {}
    for name, content in texts.items():
        results[name] = _full_analysis(content)

    combined_words = sum(r['word_count'] for r in results.values())
    combined_sentences = sum(r['sentence_count'] for r in results.values())

    output = {
        'files_analysed': len(results),
        'total_words': combined_words,
        'total_sentences': combined_sentences,
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'analyses': results,
    }

    os.makedirs('/result', exist_ok=True)
    with open('/result/text_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)


def action_generate_frequency_report() -> None:
    """IntermediateResult chaining: read analysis JSON, write CSV + HTML report."""
    result_dir = _resolve_env_path('ANALYSIS')
    if not result_dir:
        raise RuntimeError('ANALYSIS not set — invoke via an IntermediateResult reference')

    # Find text_analysis.json anywhere in the result tree
    analysis_path = None
    for root, _, files in os.walk(result_dir):
        if 'text_analysis.json' in files:
            analysis_path = os.path.join(root, 'text_analysis.json')
            break
    if analysis_path is None:
        raise FileNotFoundError(f'text_analysis.json not found under {result_dir}')

    with open(analysis_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs('/result', exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    # Aggregate word frequencies across all analysed files
    combined_freq: Dict[str, int] = {}
    for analysis in data.get('analyses', {}).values():
        for word, count in analysis.get('word_frequencies', {}).items():
            combined_freq[word] = combined_freq.get(word, 0) + count
    sorted_freq = sorted(combined_freq.items(), key=lambda x: x[1], reverse=True)

    # Write word_frequencies.csv
    with open('/result/word_frequencies.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'frequency'])
        writer.writerows(sorted_freq)

    # Write report.html
    top50 = sorted_freq[:50]
    max_count = top50[0][1] if top50 else 1
    bars = ''.join(
        f'<tr><td style="padding:2px 8px">{w}</td>'
        f'<td><div style="background:#4a90d9;height:16px;width:{int(c/max_count*300)}px"></div></td>'
        f'<td style="padding:2px 8px">{c}</td></tr>'
        for w, c in top50
    )
    # Per-file summary table
    file_rows = ''.join(
        f'<tr><td>{fname}</td><td>{a["word_count"]}</td><td>{a["sentence_count"]}</td>'
        f'<td>{a["readability_score"]}</td><td>{a["sentiment"]}</td></tr>'
        for fname, a in data.get('analyses', {}).items()
    )
    html = (
        '<!DOCTYPE html><html><head><title>Text Analysis Report</title></head><body>'
        '<h1>Text Analysis Report</h1>'
        f'<p>Generated: {ts}</p>'
        f'<p>Files analysed: {data.get("files_analysed", 0)} | '
        f'Total words: {data.get("total_words", 0)} | '
        f'Total sentences: {data.get("total_sentences", 0)}</p>'
        '<h2>Per-file Summary</h2>'
        '<table border="1"><tr><th>File</th><th>Words</th><th>Sentences</th>'
        '<th>Readability</th><th>Sentiment</th></tr>'
        f'{file_rows}</table>'
        '<h2>Top 50 Word Frequencies</h2>'
        f'<table>{bars}</table>'
        '</body></html>'
    )
    with open('/result/report.html', 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ACTIONS = {
    'count_words':               action_count_words,
    'count_sentences':           action_count_sentences,
    'extract_keywords':          action_extract_keywords,
    'compute_sentiment':         action_compute_sentiment,
    'compute_readability':       action_compute_readability,
    'get_text_stats':            action_get_text_stats,
    'analyze_text_file':         action_analyze_text_file,
    'generate_frequency_report': action_generate_frequency_report,
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
