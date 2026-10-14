# Knowledge-Driven IoT Fault Diagnosis Assistant

An AI copilot that watches simulated factory equipment telemetry, detects
abnormal readings, and explains the likely cause in plain English — grounded
in the equipment's own fault-reference documentation via RAG, with a citation,
not a guess.

Built for a college "Startup" subject project. See `CLAUDE.md` for how this
repository is built with Claude Code, and `docs/` for the full brief,
architecture, roadmap and tech-stack decisions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env              # then fill in your EMQX Cloud credentials
```

See `docs/03_ROADMAP.md` for the build order — this repo is meant to be built
phase by phase with Claude Code, not all at once.

## Status

Scaffold only — no application code yet. Phase 1 (project & DevOps setup) is
the first thing to build.
