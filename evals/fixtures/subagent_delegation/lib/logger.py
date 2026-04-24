"""Structured event logging. Emits JSON lines to stderr so log
aggregation pipelines can consume the stream without parsing text."""
import json
import sys


def log_event(level: str, message: str, **fields):
    line = {"level": level, "message": message, **fields}
    sys.stderr.write(json.dumps(line) + "\n")
