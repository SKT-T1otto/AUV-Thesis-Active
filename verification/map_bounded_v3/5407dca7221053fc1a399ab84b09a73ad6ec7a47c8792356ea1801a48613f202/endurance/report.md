# Public evidence endurance

Natural and injected-depletion runs are separate. Failures and censored references remain visible.
Timings measure inference only; forced faults do not estimate real-world recovery frequency.

| Arm | Mode | Batches | Complete | Early p95 s | Late p95 s | Error |
|---|---|---:|---|---:|---:|---|
| reference_v2 | natural | 184 | False | 0.33065 | 2.38537 | wall_clock_budget |
| full_w64 | natural | 640 | True | 0.01690 | 0.02385 | None |
| full_w64 | forced_depletion | 640 | True | 0.02413 | 0.01864 | None |
| full_w32 | natural | 640 | True | 0.01843 | 0.01994 | None |
| full_w32 | forced_depletion | 640 | True | 0.01897 | 0.02031 | None |
| full_w128 | natural | 640 | True | 0.02074 | 0.01840 | None |
| full_w128 | forced_depletion | 640 | True | 0.01851 | 0.01818 | None |
| no_boundary_w64 | natural | 640 | True | 0.01805 | 0.01877 | None |
| no_boundary_w64 | forced_depletion | 640 | True | 0.01853 | 0.02210 | None |
| no_local_recovery_w64 | natural | 640 | True | 0.02015 | 0.02028 | None |
| no_local_recovery_w64 | forced_depletion | 193 | False | 0.01736 | 0.01945 | local_recovery_disabled_ablation |
| window_map_w64 | natural | 640 | True | 0.02073 | 0.02448 | None |
| window_map_w64 | forced_depletion | 640 | True | 0.02133 | 0.02013 | None |
