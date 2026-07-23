"""The autopilot: a self-checking daily loop over the whole pipeline.

Pulls new data, validates it, matures and re-gates samples through the
promotion ladder, and hands a day-over-day briefing to a human via Obsidian and
Telegram. See ``docs/specs/2026-07-22-autopilot-design.md``.
"""
