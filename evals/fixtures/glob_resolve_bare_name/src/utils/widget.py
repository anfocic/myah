"""Widget rendering helpers. Buried deep inside src/utils/ on purpose
— the eval task gives the model only the bare filename `widget.py`
and measures whether it uses `glob` to resolve the full path before
calling `read_file`, rather than guessing at paths."""


def render_widget(name: str, payload: dict) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in payload.items())
    return f"<widget name={name!r} {attrs}/>"


def parse_widget(xml: str) -> dict:
    body = xml.strip().removeprefix("<widget ").removesuffix("/>")
    out: dict[str, str] = {}
    for pair in body.split():
        if "=" not in pair:
            continue
        k, _, v = pair.partition("=")
        out[k] = v.strip('"').strip("'")
    return out


def widget_names(widgets: list[str]) -> list[str]:
    return [parse_widget(w).get("name", "") for w in widgets]
