#!/usr/bin/env python3
"""
Datetime Package for Brane.

The action name arrives as sys.argv[1] (set via container.yml command.args).
Input arguments arrive as JSON-encoded uppercase env vars.
Output is printed to stdout as:  output: <value>
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_local() -> datetime:
    """Return the current local datetime with timezone info."""
    return datetime.now(tz=datetime.now(timezone.utc).astimezone().tzinfo)


def _env(name: str) -> str:
    """Read a JSON-encoded env var and return it as a plain string."""
    raw = os.environ.get(name.upper(), '""')
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return str(value)


def _out(value: str) -> None:
    """Print the Brane output line and flush.
    json.dumps ensures the value is always a quoted YAML string,
    so serde_yaml never misparses numbers or timestamps as non-string types."""
    print(f"output: {json.dumps(str(value))}", flush=True)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def get_iso() -> None:
    """ISO 8601 with UTC offset: 2026-06-14T22:58:17+02:00"""
    _out(_now_local().isoformat(timespec='seconds'))


def get_date() -> None:
    """Date only: 2026-06-14"""
    _out(datetime.now().strftime('%Y-%m-%d'))


def get_time() -> None:
    """Time only: 22:58:17"""
    _out(datetime.now().strftime('%H:%M:%S'))


def get_human() -> None:
    """Human-readable: Saturday, June 14 2026 10:58 PM"""
    _out(datetime.now().strftime('%A, %B %-d %Y %-I:%M %p'))


def get_unix() -> None:
    """UTC Unix timestamp: 1749945497"""
    _out(str(int(time.time())))


def get_formatted() -> None:
    """Custom strftime format supplied by the caller via FORMAT_STR env var."""
    fmt = _env('FORMAT_STR')
    if not fmt:
        result = json.dumps({'error': 'format_str is required', 'status': 'failed'})
        _out(result)
        sys.exit(1)
    try:
        _out(datetime.now().strftime(fmt))
    except Exception as exc:
        result = json.dumps({'error': f'Invalid format string: {exc}', 'status': 'failed'})
        _out(result)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_ACTIONS = {
    'get_iso':       get_iso,
    'get_date':      get_date,
    'get_time':      get_time,
    'get_human':     get_human,
    'get_unix':      get_unix,
    'get_formatted': get_formatted,
}


def main() -> None:
    if len(sys.argv) < 2:
        _out(json.dumps({'error': 'No action name in argv[1]', 'status': 'failed'}))
        sys.exit(1)
    action = sys.argv[1]
    handler = _ACTIONS.get(action)
    if handler is None:
        _out(json.dumps({'error': f'Unknown action: {action!r}', 'status': 'failed'}))
        sys.exit(1)
    handler()


if __name__ == '__main__':
    main()
