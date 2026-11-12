# Aparichit: Machine Unlearning & Privacy Leaderboard

Generated on: 2026-11-12 15:43:18

| Method | Forget ROUGE-L | Retain ROUGE-L | Forget Quality (p-val) | MIA ROC-AUC | Privacy Leakage | Eval Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (Memorized Full) | 0.8970 | 1.0000 | 9.8018e-85 | 0.9282 | 0.8564 | 159.8 |
| Method 2 (Fisher Masking) | 0.1846 | 0.9636 | 1.7091e-39 | 0.2179 | 0.5641 | 140.8 |
| Method 4 (Generic-Label Tier 2) | 0.5691 | 1.0000 | 9.4563e-79 | 0.9139 | 0.8279 | 150.6 |
| Method 3 (Task Vectors + TIES) | 0.2793 | 0.6333 | 3.3095e-55 | 0.8629 | 0.7258 | 8.5 |
