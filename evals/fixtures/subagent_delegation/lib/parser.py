"""Command-string parser. Splits a whitespace-separated command line
into a verb and positional arguments, respecting single-quoted tokens."""
import shlex


def parse_command(line: str) -> tuple[str, list[str]]:
    tokens = shlex.split(line)
    if not tokens:
        return "", []
    return tokens[0], tokens[1:]
