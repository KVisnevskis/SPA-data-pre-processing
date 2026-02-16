# GitHub Issue Backlog Notes

These are tracked reminders to convert into formal GitHub issues.

## 1) Stage-3 fixed sync method update in thesis
- Pipeline spec now reflects implemented behavior: fixed-run sync may use transformed phi (`none`, `invert`, `abs`, `auto`).
- Action: update thesis Chapter 3 method description to match implementation details.

## 2) Add stronger Stage-3 sync regression checks when reference values exist
- Current tests validate broad behavior/ranges, not exact expected shifts for sample runs.
- Action: once known-good reference shifts are established, add pinned assertions to `tests/test_time_sync.py`.

## 3) Documentation cleanup pass
- Current docs are acceptable for now but need a later quality pass (formatting/encoding/readability and top-level README content).
- Action: schedule a documentation-focused update before final release.
