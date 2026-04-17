# tools/bash.py
import os
import subprocess

MAX_OUTPUT_BYTES = 50_000
DEFAULT_TIMEOUT = 30


def _truncate(text: str, name: str) -> str:
    if len(text) > MAX_OUTPUT_BYTES:
        return text[:MAX_OUTPUT_BYTES] + f"\n... ({name} truncated at {MAX_OUTPUT_BYTES} chars)"
    return text


def bash(command: str, cwd: str = ".", timeout: int = DEFAULT_TIMEOUT):
    """Run a shell command, returning stdout + stderr + exit code.

    Sensitive by design: the permission layer gates every call so the user
    sees the exact command before it runs. shell=True is safe in that model
    because the "attacker" (the LLM) can't reach the shell without a human
    authorizing the string first.

    Output is capped at MAX_OUTPUT_BYTES per stream to keep a noisy command
    (`ls -R /` etc.) from nuking the context window.
    """
    resolved_cwd = os.path.expanduser(cwd)
    if not os.path.isdir(resolved_cwd):
        return f"Working directory not found: {cwd}"

    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=resolved_cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s: {command}"
    except Exception as e:
        return f"Error running command: {e}"

    stdout = _truncate(proc.stdout or "", "stdout").rstrip()
    stderr = _truncate(proc.stderr or "", "stderr").rstrip()

    parts = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    parts.append(f"exit: {proc.returncode}")
    return "\n\n".join(parts)
