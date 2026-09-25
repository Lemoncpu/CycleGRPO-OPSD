# Style and Literature Notes

The manuscript follows the observation-first, contribution-forward style visible in Xiangtai Li's conference papers and project pages: a concrete dense-perception failure, a compact unified method, and a benchmark table that ties each component to a measurable effect. The planned main-text budget is approximately 0.6 page abstract, 1.3 pages introduction, 0.8 page formulation, 2.6 pages method, 2.8 pages experiments, 0.5 page related work, and 0.4 page limitations/conclusion. Figures use restrained blue/orange accents and captions carry the interpretation.

Recent OPD motivation sources inspected:

- GKD (ICLR 2024): student-generated trajectories reduce train/inference mismatch and permit flexible teacher losses.
- MiniLLM (ICLR 2024): reverse-KL on-policy optimization avoids over-smoothing and exposure bias in generative LMs.
- DistiLLM (ICML 2024): divergence choice and student-output generation jointly determine efficiency and stability.
- DistiLLM-2 (ICML 2025 Spotlight): teacher and student generated data benefit from different, contrastive objectives.
- Hybrid Policy Distillation (ICML 2026): forward/reverse KL and off/on-policy data have complementary inductive biases.
- Revisiting OPD (2026 preprint, not treated as an accepted-paper citation): sampled token supervision can be fragile under prefix drift, motivating reliability controls.

These sources motivate dense on-policy supervision, but Pixel-OPSD differs by using pixel IoU and privileged spatial evidence in a pixel-level MLLM cycle. Claims about accepted venues are limited to the conference metadata verified in the corresponding proceedings or paper front matter.

Conference-version reading update:

- GAR / ICLR 2026: 29-page camera-ready, five main sections (Introduction, Related Work, Method, Experiments, Conclusion), with an early qualitative teaser and benchmark-driven method subsections.
- MMaDA-Parallel / ICLR 2026: 35-page camera-ready, five main sections and a problem-observation-first introduction followed by a unified method and large ablation tables.
- RMP-SAM / ICLR 2025: 23-page camera-ready, six main sections including a dedicated task/problem definition before method and experiments.
- OMG-Seg / CVPR 2024: 12-page official CVF version; SAMTok / CVPR 2026: 12-page official CVF version.
- Open-o3-Video / ICML 2026: official ICML poster and abstract page verified, but no conference paper PDF was exposed by the current proceedings page.
- Video K-Net / CVPR 2022: official IEEE DOI 10.1109/CVPR52688.2022.01828 and author record verified; the accessible CVF/IEEE PDF endpoint was not available in this environment.

The draft therefore uses the author's recurring structure (observation, unified method, broad table, concise conclusion) while keeping the ICLR 2027 submission within the 9-page main-text limit.
