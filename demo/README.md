# Demo data

This folder contains a minimal public demo for checking the expected input format.

Contents:

- `images/demo_multikernel_rgb.jpg`: one metadata-stripped RGB multi-kernel image resized for lightweight distribution.
- `examples/Healthy/healthy_001.png`: one representative healthy kernel crop.
- `examples/Diseased/diseased_001.png`: one representative diseased kernel crop.
- `demo_plot_predictions.csv`: a small example table showing the plot-level prediction schema expected by analysis scripts.

The demo data are for pipeline inspection and format verification only. They are not the full evaluation set and should not be used to reproduce manuscript-level performance metrics.

If a trained checkpoint has been placed under `output_results/MoPE_Ranking/`, or if `examples/Healthy/` and `examples/Diseased/` have been populated with sufficient training crops, the demo image can be processed with:

```bash
python few_shot_cellpose_mope_ranking.py --images demo/images/demo_multikernel_rgb.jpg
```
