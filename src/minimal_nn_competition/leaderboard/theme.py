"""Shared visual tokens and common styles for generated leaderboard pages."""

FONT_FAMILIES = {
    "display": 'ui-serif, Georgia, "Times New Roman", serif',
    "body": 'system-ui, -apple-system, "Segoe UI", sans-serif',
    "mono": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
}

LIGHT_COLORS = {
    "bg": "#f6f8f4",
    "surface": "#ffffff",
    "ink": "#18231d",
    "muted": "#5c6a60",
    "rule": "#dce5da",
    "rule-strong": "#aab8a9",
    "hover": "#f1f6e8",
    "pass": "#23683f",
    "fail": "#963f3b",
    "pass-tint": "#e7f3e9",
    "fail-tint": "#faedeb",
    "focus": "#355f3a",
    "accent-snake": "#276b63",
    "accent-image-compression": "#8b5a0b",
    "accent-chess": "#345d91",
    "accent-control": "#7041ad",
    "accent-cta": "#c3d0db",
    "accent-cta-hover": "#b0c1ce",
    "accent-cta-ink": "#1b2a36",
    "chart-obstacle": "#818c84",
    "chart-danger": "#dc4843",
    "chart-snake-body": "#4fae67",
}

DARK_COLORS = {
    "bg": "#101512",
    "surface": "#161f18",
    "ink": "#edf3ea",
    "muted": "#a4b2a2",
    "rule": "#28362b",
    "rule-strong": "#465745",
    "hover": "#1c2a1c",
    "pass": "#7bd49a",
    "fail": "#f0988e",
    "pass-tint": "#17261c",
    "fail-tint": "#2a201e",
    "focus": "#a6db72",
    "accent-snake": "#84d2bd",
    "accent-image-compression": "#e9bd5c",
    "accent-chess": "#a2c1eb",
    "accent-control": "#c3a1f5",
    "accent-cta": "#879bad",
    "accent-cta-hover": "#9aaaba",
    "accent-cta-ink": "#14212b",
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
