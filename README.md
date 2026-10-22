# python-template

A minimal template for Python projects.

It uses Python 3.12, `uv`, `ruff`, `ty`, `pytest`, and optional agent support for
AI-assisted development.

## Quickstart

```bash
uv sync --dev
uv run pytest
```

## Development Commands

```bash
uv run ruff format
uv run ruff check
uv run ty check
uv run pytest
uv build
```

## What Is Included

- `src/` package layout
- `uv` dependency management
- `ruff` formatting and linting
- `ty` type checking
- `pytest` test discovery
- GitHub Actions CI
- Generic agent instructions in `AGENTS.md`
- Optional OpenCode config, agents, and skills under `.opencode/`

## Optional Agent Support

This template works as a normal Python project without OpenCode. If you use
AI coding agents, the repository also includes:

- `opencode.json` with conservative default permissions and useful commands
- `.opencode/agents/python-engineer.md` for focused Python implementation work
- `.opencode/agents/research.md` for read-only academic research (CrossRef/OpenAlex/PubMed MCPs, disabled by default)
- `.opencode/skills/dev-workflow` for planned, verified changes
- `.opencode/skills/grill-me` to stress-test a plan before implementation
- `.opencode/skills/code-review` for actionable code reviews
- `.opencode/skills/session-retro` to capture lessons and follow-ups
- `.opencode/skills/caveman` for terse technical communication
- `.opencode/skills-available/brooks/` with the opt-in, explicit-invocation brooks-lint skills

To enable brooks, copy `.opencode/skills-available/brooks/skills/*` into
`.opencode/skills/`.
