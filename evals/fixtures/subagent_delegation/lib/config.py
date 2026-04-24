"""Runtime configuration defaults. Read once at process start; values
can be overridden via the `APP_*` environment variables."""

DEFAULTS = {
    "max_connections": 100,
    "request_timeout_s": 30,
    "cache_capacity": 512,
    "log_level": "INFO",
}
