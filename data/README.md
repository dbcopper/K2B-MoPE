# Data layout

This code release does not include raw images, private fieldbooks, trained checkpoints, or generated result tables.

To reproduce the full manuscript analyses, place released data and prediction tables under the following layout.

```text
data/
  raw_images/
  metadata/
  seeds_png/
    health/
    disease/
    test/

examples/
  Healthy/
  Diseased/

results/
  result1/
  result2/
    per_kernel/
      table8_per_kernel_predictions.csv
    per_plot/
      table8_per_plot_predictions.csv
    models/
      threshold_info.json
  result3/
    per_plot/
      mope_ablation_per_plot_predictions.csv
```

`data/seeds_png/` is the default input root for `scripts/run_newdata_only_seed_models.py`.

## Required plot-level prediction columns

`results/result2/per_plot/table8_per_plot_predictions.csv` should contain:

```text
year
plot
line
manual_vsk_pct
svm_predicted_vsk_pct
rf_predicted_vsk_pct
resnet18_predicted_vsk_pct
vit_predicted_vsk_pct
mope_predicted_vsk_pct
```

## Required kernel-level prediction columns

`results/result2/per_kernel/table8_per_kernel_predictions.csv` should contain at least:

```text
year
plot
line
kernel_id
mope_score
manual_vsk_pct
```

Additional method-specific score columns can be included for comparison scripts.

## Independent evaluation split

The manuscript evaluation split contains 80 plot-level images:

- 31 images from 2024.
- 49 images from 2025.

The corrected plot identifier is `15131`; stale transposed plot identifiers should not be used.
