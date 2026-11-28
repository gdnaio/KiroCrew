# A Codex runtime's kill deletes its private database directory once it confirms the runtime's processes are gone

Decided by: Bolin Chen (@bolichen97), maintainer
Date: 2026-11-28

## Decision

When `AcpRuntime.kill` confirms a Codex runtime's process tree dead, it deletes that runtime's private SQLite home (`CODEX_SQLITE_HOME`) at once through a one-directory mode of `agent_scratch.sweep_dead_scratch` that keeps every check of the sweep but its idle wait, and leaves the runtime's own scratch directory and every death the kill does not confirm to the hourly sweep.

## Why

- The directory is mounted by that one process alone and no other process reads it once the process is dead; config, auth and thread rollouts stay in `CODEX_HOME`, so deleting it at the kill removes nothing the hourly sweep would not delete under the same checks an hour later.
- Each private index can be about 2.25 GiB, and the hourly sweep reclaims a dead one only after an hour idle, so short-lived runtimes (initialize-timeout retries, background jobs) piled up dead directories between sweeps: one reported host had 242 of them, 252.7 GiB, in about 82 minutes.
- The runtime's own scratch directory is the session tree's work directory, which other processes mount, so it stays with the sweep alone.

## Evidence

- Maintainer decision: https://github.com/kirodotdev/KiroCrew/issues/42024#issuecomment-7238457431
- Issue: https://github.com/kirodotdev/KiroCrew/issues/42024
