# FDG implementation correction

The former public gate documentation described a single-temperature FDG rule,
but its executable gate used `Linear(25, 3)`. The corrected core implements
the revised manuscript equations directly:

- Normalization uses the population mean and standard deviation of all pooled
  Healthy and Diseased training kernels. The effective inference denominator is
  `std + 1e-10`. Evaluation kernels do not enter these statistics.
- Deviation is the mean absolute standardized value within each 11/11/3-feature
  group. A softmax of these three values divided by one positive learned
  temperature provides the gate. Temperature starts at 1.0 and is represented
  by a softplus parameter plus a small positive floor.
- The fixed primary schedule is Adam, learning rate 0.001, weight decay 0.0001,
  16 epochs, batch size 128 pairs, margin 0.3 and seed 42. Kernel-level LOO
  training is not run automatically because kernels share parent images.
- The image threshold is `mu_d + 0.25 * (image_mean - (mu_h + mu_d) / 2)`.
  Each kernel at or above the threshold contributes equally to the VSK-style
  percentage. The score is an ordinal representation, not calibrated biological
  disease severity.
- The core segmentation default is the named Cellpose `cyto` model and a
  diameter of 95 pixels. New segmentation requires `cellpose>=3.1,<4`; v4 is
  rejected because its API ignores the old `model_type` argument. All active
  image-training and inference paths share the explicit v3 constructor.
  Cellpose remains optional until image segmentation is actually requested.
  New mask cache names identify the model and diameter, so earlier caches
  without that identity are not silently reused.

The model/version guard follows the official [Cellpose 3 model documentation](https://cellpose.readthedocs.io/en/v3.1.1.1/models.html)
and [Cellpose 4 constructor source](https://cellpose.readthedocs.io/en/latest/_modules/cellpose/models.html).

## Checkpoint and output handling

The schema is `fdg_pooled_reference_v1`, with normalization reference
`all_training`. Newly saved checkpoints record training, gating, segmentation
and aggregation settings. The loader rejects unversioned linear gates,
healthy-only schemas, non-finite or mismatched normalization arrays and legacy
gate keys. The primary loader also rejects a different aggregation rule.

No old checkpoint has been converted. Corrected runs use a separate output
directory. Existing evaluation tables, figures and trained weights are preserved.

## Result provenance

Synthetic CPU checks establish the implementation contract and save/load
compatibility. They do not establish the 80-sample evaluation performance.
The previously reported MAE 6.600 and other manuscript values have not been
regenerated or validated by this code correction. A new frozen checkpoint,
per-kernel predictions, per-image predictions and analysis outputs are required
before those values can be attributed to this implementation.

The repository currently provides code and a minimal demo. It does not provide
the complete raw image dataset, private fieldbooks, trained weights or generated
evaluation results. The manuscript's data-availability statement must reflect
the materials actually deposited, including any separate repository record.
