# tools/utils.py
from datetime import datetime

from tools.spec import register


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


register(
    name="get_current_time",
    description="Returns the current date and time",
    adapter=lambda _args, _cwd: get_current_time(),
    read_only=True,
)
