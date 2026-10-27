"""Shared visual tokens and common styles for generated leaderboard pages."""

FONT_FAMILIES = {
    "display": 'ui-serif, Georgia, "Times New Roman", serif',
    "body": 'system-ui, -apple-system, "Segoe UI", sans-serif',
    "mono": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
}

LIGHT_COLORS = {
    "bg": "#f4f6f4",
    "surface": "#fbfcfa",
    "ink": "#17231f",
    "muted": "#63706b",
    "rule": "#d9e0db",
    "rule-strong": "#aebbb5",
    "hover": "#edf2ef",
    "pass": "#1f6b43",
    "fail": "#8a3434",
    "pass-tint": "#e6f1ea",
    "fail-tint": "#f8e9e7",
    "focus": "#2f5f57",
    "accent-snake": "#256b66",
    "accent-image-compression": "#8a5b16",
    "accent-chess": "#3f5f86",
    "accent-control": "#6f42c1",
    "chart-obstacle": "#8a9a94",
    "chart-danger": "#e5484d",
    "chart-snake-body": "#57b47a",
}

DARK_COLORS = {
    "bg": "#0d1513",
    "surface": "#131d1a",
    "ink": "#e7efeb",
    "muted": "#9aaba4",
    "rule": "#273631",
    "rule-strong": "#40534c",
    "hover": "#182622",
    "pass": "#74cf9b",
    "fail": "#f09b91",
    "pass-tint": "#15251d",
    "fail-tint": "#2b1e1c",
    "focus": "#8bcacb",
    "accent-snake": "#7fcac1",
    "accent-image-compression": "#e0b76b",
    "accent-chess": "#9bb9df",
    "accent-control": "#b392f0",
    "chart-obstacle": "#61766d",
    "chart-danger": "#ff827a",
    "chart-snake-body": "#91dfb0",
}


def _css_declarations(values):
    return "\n".join(f"  --{name}: {value};" for name, value in values.items())


SITE_THEME_CSS = f"""
:root {{
  color-scheme: light dark;
{_css_declarations(LIGHT_COLORS)}
  --display: {FONT_FAMILIES['display']};
  --body: {FONT_FAMILIES['body']};
  --mono: {FONT_FAMILIES['mono']};
  --chart-snake-head: var(--pass);
  --chart-boundary: var(--rule-strong);
  --chart-cart: var(--ink);
  --chart-muted: var(--muted);
}}
@media (prefers-color-scheme: dark) {{
  :root {{
{_css_declarations(DARK_COLORS)}
  }}
}}
"""

SITE_BASE_CSS = """
*, *::before, *::after { box-sizing: border-box; }
[hidden] { display: none !important; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--body);
  font-size: 16px;
  line-height: 1.55;
}
h1, h2, h3 { font-family: var(--display); }
a { color: var(--accent-snake); text-underline-offset: 3px; }
a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 3px; }
code { font-family: var(--mono); }
"""

AUXILIARY_TABLE_CSS = """
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td {
  border-bottom: 1px solid var(--rule);
  padding: 0.5rem;
  text-align: left;
}
th {
  color: var(--muted);
  font: 700 10px/1.4 var(--mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.muted, .failed { color: var(--muted); }
"""
