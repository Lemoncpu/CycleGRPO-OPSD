# Pixel-OPSD LaTeX project

This directory is the complete LaTeX source corresponding to the supplied
paper archive.  `main.tex` now includes the completed qualitative comparison
as `figures/qualitative_comparison.pdf`.

Figure 4 was generated from the audited GroundingSuite official-training-long-
prompt outputs using the following matched runs:

- SAMTok base model: `logs/evaluation/Qwen3-VL-4B-SAMTok`
- CycleGRPO post-training comparator: `logs/evaluation/CycleGRPO-4B`
- Pixel-OPSD: `step_178`
- Three examples retained from the reviewed set (the third, fifth, and sixth rows)

The panel uses the same manifest for all three models and displays the input
caption, method names, aligned equal-width panels with semi-transparent mask
overlays and 0.65pt boundaries, and per-model IoU values. Source photographs
fill each panel without padding frames and are embedded at high PDF resolution
without interpolation. It contains
no internal case IDs or extra explanatory text; every selected Pixel-OPSD IoU
is more than 0.1 above both baselines.

The figure generator is `make_qualitative_figure.py`.  Its defaults use the
local repository paths; pass `--ours-dir`, `--baseline-dir`, `--cycle-dir`,
`--image-root`, and `--output` when reproducing the figure elsewhere.  The generated PDF is
already included, so Overleaf can compile the project without the evaluation
artifacts.

Build locally with:

```text
./build_pdf.sh
```
