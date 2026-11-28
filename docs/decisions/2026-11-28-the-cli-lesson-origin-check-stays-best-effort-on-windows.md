# The CLI lesson origin check stays best-effort on Windows; no Windows agent signal is built

Decided by: Bolin Chen (maintainer, @bolichen97)
Date: 2026-11-28

## Decision

With `memory.lesson_origin_gate` at `enforce`, `kirocrew learn add` keeps telling an agent's shell
from a person's terminal the way it does today, which on Windows means only the environment Kiro
Crew gives an agent's processes; the help text and docs say the check is best-effort, and no
Windows-only agent signal is built.

## Why

- The check is a label that ranks an agent's `learn_add` below a person's rules, not a fence, and
  the setting is off by default. On Windows Kiro Crew has no sandbox of its own around an agent and
  no confinement it can read (a Kiro CLI session may run in that tool's own sandbox, which Kiro Crew
  cannot verify), so any extra Windows signal would still only add evidence while costing
  Windows-only process code.
- It is best-effort on every platform, not only on Windows: on Linux a process an agent starts
  through the user's service manager also runs outside the agents cgroup, a documented residual.
  Saying so in the help text is the honest state.

What a person on Windows with `enforce` on can do:

- Keep `enforce` on: it still catches an agent that runs `kirocrew learn add` as is.
- Read `kirocrew learn list` now and then for rules you did not write, and remove them with
  `kirocrew learn remove`.

## Evidence

- https://github.com/kirodotdev/KiroCrew/issues/42579 -- the report, and the issue's own suggestion
  to state the Windows limit in the help text and the memory spec.
- https://github.com/kirodotdev/KiroCrew/issues/42579#issuecomment-7238651519 -- Bolin Chen's
  decision on the issue.
