# Same-input inference diagnostic matrix

Synthetic public rays and detector trajectories are independently specified. Internal faults test mechanisms, not physical consistency, natural recovery frequency or task performance.

| Arm | Mode | Attempted | Complete | Expected negative | Check | Error |
|---|---|---:|---|---|---|---|
| full_w32 | natural | 384 | True | False | True |  |
| full_w32 | target_depletion | 384 | True | False | True |  |
| full_w32 | geometry_depletion | 384 | True | False | True |  |
| full_w64 | natural | 384 | True | False | True |  |
| full_w64 | target_depletion | 384 | True | False | True |  |
| full_w64 | geometry_depletion | 384 | True | False | True |  |
| full_w128 | natural | 384 | True | False | True |  |
| full_w128 | target_depletion | 384 | True | False | True |  |
| full_w128 | geometry_depletion | 384 | True | False | True |  |
| window_map_w64 | natural | 384 | True | False | True |  |
| no_boundary_w64 | target_depletion | 384 | True | False | True |  |
| no_local_recovery_w64 | target_depletion | 193 | False | True | True | local_recovery_disabled_ablation |
