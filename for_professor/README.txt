REVISION RESULTS PACKAGE — drones-4250454
==========================================
Generated: 2026-05-02

This folder contains all updated results and new sensitivity experiments
produced in response to the reviewer comments.

────────────────────────────────────────────────────────────────────────
figures/
────────────────────────────────────────────────────────────────────────

fig2_RDM.pdf
    Learning curves for all methods on RDM map (original paper figure).

fig3_RBM.pdf
    Learning curves for all methods on RBM map (original paper figure).

lambda_sensitivity_heatmap.pdf             [Reviewer 3, Item 4]
    Heatmap of victims localised (out of 10) across a 3×5 grid of
    reward coefficients (lambda_thr × lambda_new). FedQMIX, seed 1,
    3000 episodes. Performance stays between 8.3–10.0 on both maps,
    confirming the reward design is not critically sensitive to the
    exact coefficient values.

partial_fedavg_robustness.pdf              [Reviewer 2, Item 7]
    Mean ± std victims localised vs. worker drop probability (0–70%).
    FedQMIX, seed 1, 3000 episodes. Performance never drops below 9.2/10
    even at 70% packet loss, demonstrating robustness of FedAvg
    aggregation under intermittent connectivity.

channel_shift_robustness.pdf               [Reviewer 1, Item 5]
    Victims localised vs. SNR offset applied at evaluation time (dB).
    Policy trained at nominal SNR; tested under increasing channel
    mismatch. Graceful degradation: −5 dB still yields 50–64% of victims;
    complete collapse only at −20 dB (far outside realistic urban variation).

────────────────────────────────────────────────────────────────────────
tables/
────────────────────────────────────────────────────────────────────────

seed_table_RBM.tex / seed_table_RDM.tex
    Per-seed evaluation scores (last 5 checkpoints) for all methods.
    Corresponds to Table 3 in the paper.

metrics_summary_ci.tex
    Aggregated mean ± 95% CI across 5 seeds. Includes:
    • victims_localised   (main metric)
    • success_rate        = victims / 10      [Reviewer 2, Item 1]
    • time_to_first_detection (steps)         [Reviewer 2, Item 3]
    • energy_per_victim
    • throughput / reward

significance_tests.tex
    Wilcoxon / Mann-Whitney U pairwise significance tests between methods.

tableC_sensitivity.tex
    Robustness to BLE beacon intermittency (p_on) and SNR threshold
    perturbations (original sensitivity analysis from the paper).

table_lambda_sensitivity.tex               [Reviewer 3, Item 4]
    LaTeX table version of the lambda heatmap.

table_partial_fedavg.tex                   [Reviewer 2, Item 7]
    LaTeX table: victims mean ± std for each drop probability × map.

table_channel_shift.tex                    [Reviewer 1, Item 5]
    LaTeX table: victims mean ± std under each SNR offset × map.

────────────────────────────────────────────────────────────────────────
data/
────────────────────────────────────────────────────────────────────────

seed_table_RBM.csv / seed_table_RDM.csv
    Raw per-seed evaluation scores (source for the seed tables).

metrics_summary_ci.csv
    Aggregated metrics with confidence intervals across all seeds.

sensitivity_lambda_{RBM,RDM}.csv
    Raw lambda sensitivity sweep results.

sensitivity_partial_fedavg_{RBM,RDM}.csv
    Raw partial FedAvg sweep results.

sensitivity_channel_shift_{RBM,RDM}.csv
    Raw channel-shift robustness results.

────────────────────────────────────────────────────────────────────────
NOTES
────────────────────────────────────────────────────────────────────────

• Success rate (Reviewer 2, Item 1) is victims_localised / 10.
  It is already embedded in metrics_summary_ci.csv and .tex.
  FED achieves 99.2% on RBM and 99.8% on RDM (best of all methods).

• Time-to-first-detection (Reviewer 2, Item 3) is also in
  metrics_summary_ci.csv. RBM values are 0 steps because the map
  is compact enough that UAVs reach BLE range immediately.
  IPPO TTF is not available (was not tracked in original IPPO runs).

• PSO localisation error (Reviewer 2, Item 2) applies only to runs
  with the SLAL surrogate (model=True). The paper's main FED method
  uses model=False; loc_error is available for the MOD method only
  if re-run with updated code.

• All three new sensitivity experiments use FedQMIX, seed 1,
  3000 training episodes — matching the main paper configuration.
