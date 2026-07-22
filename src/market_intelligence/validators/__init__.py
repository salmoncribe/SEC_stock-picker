"""Validation helpers.

Source-specific validators live in ``sec.py`` / ``fred.py``. They mutate a
record's ``validation_status`` / ``validation_errors`` (never raise for data
problems) so that anomalies are recorded rather than silently dropped.
"""

from __future__ import annotations
