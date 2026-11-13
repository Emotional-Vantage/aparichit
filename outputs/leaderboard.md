# Aparichit: Machine Unlearning Three-Way Benchmark Leaderboard

Generated on: 2026-11-13 06:40:47

### Evaluation Overview:
- **Forget Set**: 400 QA pairs target for unlearning
- **Retain Set**: 400 QA pairs active retain subset
- **Do-Not-Train Set**: 200 QA pairs unseen control strictly excluded from all training
- **References**: Comparison against both $M_{full}$ and separately trained $M_{retain}$

| Model / Method | Forget Loss | Retain Loss | DNT Loss | Forget ROUGE-L | Retain ROUGE-L | Delta Forget vs M_retain | Delta Retain vs M_retain | MIA Raw AUC | MIA Adj AUC | Unlearn Time | Eval Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reference: Pretrained (M_base) | 2.819 | 2.928 | 2.836 | 0.193 | 0.162 | -1.250 | +2.895 | 0.506 | 0.506 | N/A | 471.7s |
| Baseline: Memorized Full (M_full) | 0.042 | 0.029 | 3.978 | 0.870 | 0.927 | -4.027 | -0.004 | 0.998 | 0.998 | N/A | 376.1s |
| Reference: Retain-Only Model (M_retain) | 4.069 | 0.033 | 4.109 | 0.273 | 0.927 | +0.000 | +0.000 | 0.502 | 0.502 | N/A | 376.6s |
| Method 2: Fisher Retain-Anchored (M_unlearned_fisher) | 9.113 | 0.013 | 4.994 | 0.237 | 0.963 | +5.044 | -0.020 | 0.349 | 0.651 | 120.0s | 316.6s |
| Method 1: Gradient Difference (M_unlearned_gd) | 10.324 | 0.015 | 5.479 | 0.195 | 0.961 | +6.255 | -0.018 | 0.276 | 0.724 | 93.1s | 323.5s |
| Method 4: Generic-Label Tier 1 (M_unlearned_generic_t1) | 0.337 | 0.008 | 4.350 | 0.751 | 0.972 | -3.732 | -0.025 | 0.999 | 0.999 | 73.1s | 362.5s |
| Method 3: Magnitude-Trimmed Task Vector (M_unlearned_tv) | 3.482 | 1.879 | 3.851 | 0.119 | 0.130 | -0.587 | +1.846 | 0.638 | 0.638 | 175.8s | 462.9s |
