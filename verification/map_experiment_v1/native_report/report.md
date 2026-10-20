# OLD / NEW maps with independently adapted ZERO, B2 and B3

All scheduled native policies completed; this does not assert a performance improvement.

| Group | Policies | Found | Success | Collision | Penalized found steps |
|---|---:|---:|---:|---:|---:|
| OLD_B2 | 1 | 0.0000 | 0.0000 | 1.0000 | 8.00 |
| NEW_B2 | 1 | 0.0000 | 0.0000 | 1.0000 | 8.00 |
| OLD_B3 | 1 | 0.0000 | 0.0000 | 1.0000 | 8.00 |
| NEW_B3 | 1 | 0.0000 | 0.0000 | 1.0000 | 8.00 |

- Paired deltas are NEW minus OLD. Negative is better for collision, time and cost; positive is better for Found and Success.
- Learning repeats are training seeds; shared evaluation scenes are a separate sampling axis.
- One-seed intervals contain scene uncertainty only. ZERO has no training repeats.
- Core planning timing covers successful proposals; runtime timing includes environment and actor. Report fallback frequency as well.
- Native evaluation only: each model is evaluated with its own training map. No historical weights are restored or relabelled by preparation.
- Source/config/checkpoint hashes and final-episode selection are retained. Outputs are never overwritten.
