# Bounded-validation transport correction — 2026-10-27

Parent commit:b57d8e5c9f776c912138a71dc613e3bfa4f29fa4.
Current source:b7fa87913db588f43ccb1c94e278a169be48777775658bb1a4867bf11a2cdcbe. Verification-only evolution.
All parent runtime bytes remain unchanged except reviewed source-admission
metadata. No controller,RL,reward,dynamics,task dimensions or termination change.
Original safety-learning findings3/4/6 remain unresolved and admission remains suspended.

## Reported Linux failure

The parent checkout/package/source passed;363 tests passed in320.421 seconds.
Five native cases completed32 physical steps each without collision before the
aggregate900-second wall deadline killed the child. The two400-step research
evaluations had not started. Completed cases were baseline0002,repeat0002,
nominal composition0002,rule0002 and rule0006. Episode timeout at32 steps is
the bounded horizon; it differs from the parent process900-second timeout.
The failed directory is never overwritten or changed to passed.

## Transport correction

Stdout,stderr and unittest output are saved and displayed live. The parent
prints a heartbeat while waiting. Native cases show every8 physical steps.
Default check wall timeout is3600 seconds; explicit checks-timeout may select
up to7200 seconds. This changes only wall-clock waiting,never the8x32 physical
budget or update/checkpoint/training restrictions. Timeout retains partial logs
and stops our child process group on Linux.

## Explicit prefix continuation

An explicit safe-rl-prefix accepts only the exact completed five-case b57d8e5
timeout prefix under this checkout work directory. It rechecks the host,
Python/libraries,source identity,363 parent tests,reservations,full frames,
exact same-host signature repeat,nominal numerical parity and safety-case
collision/motion acceptance. Every borrowed file is hashed; the old source
identity is retained in each borrowed record. Current regressions still run.
Only fault_all_four and buffer_exit_all_four are then initialized,from their
original independent reset inputs. There is no checkpoint or state restoration.
The total ledger is five inherited plus two new starts,160 plus64 steps.

A separate one-use claim prevents repeating the same continuation and silently
resetting its budget. If a continuation fails,inspect its fresh outputs;do not
delete its claim. Parent files are rechecked before publishing completion.

Current verification:382 regression tests. New transport tests use
recorded parent Windows trace data in an explicitly synthetic timeout wrapper,
mocked suffix factories,and short Python process fixtures. Actual new physical
environment initializations:0;physical steps:0;checkpoint restores:0.
Algorithm mechanism evidence is explicitly referenced at its immutable parent
identity,whose runtime bytes are protected. No current native rerun or actual
Linux completion is claimed. Linux commands are given directly in Codex.
