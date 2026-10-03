# The Platonic Brain: Representational Convergence in EEG Foundation Models

Code for the paper. We test whether 14 EEG foundation models (FEMBA, LUNA, NeuroLM,
ST-EEGFormer, REVE) converge with each other, and with vision (DINOv2, V-JEPA 2, VideoMAE)
and language models (BLOOMZ, OpenLLaMA), on the CineBrain dataset. Alignment is measured
with mKNN and CKA, calibrated against a cyclic-shift null ([CYCLIC_NULL.md](CYCLIC_NULL.md)),
and compared with how well each model decodes 16 NICE EEG markers.

**Embeddings:** [huggingface.co/datasets/nicolaswaehner/platonic-embeddings](https://huggingface.co/datasets/nicolaswaehner/platonic-embeddings)
**Raw data:** [CineBrain](https://huggingface.co/datasets/Fudan-fMRI/CineBrain), Season 7.

## Pipeline

| Step | Scripts |
|---|---|
| 1. Embeddings | `src/extraction_scripts/extract_*.py` (one per model family, plus `extract_fmri.py`) |
| 2. NICE markers | `src/interp/eeg_features/eeg_features.py --subject sub-000X` |
| 3. Decodability | `src/decodability_rebuild/compute_decodability.py` |
| 4. Alignment under the cyclic null | `src/decodability_rebuild/compute_*.py`, `src/shift_tests/circular_shift_*_full.py`, `src/shift_tests_fmri/` |

The outputs of steps 3–4 are committed (`cache/`, `outputs*/` `.npz`), so the figures can be redrawn without recomputing them.

## Figures

| Paper | Script |
|---|---|
| Fig. 1 (EEG × EEG vs decodability) | `decodability_rebuild/make_all_figures.py` |
| Fig. 2 (FEMBA × DINOv2 null) | `shift_tests/plot_femba_dinov2.py` |
| Fig. 3, 5, 6 (EEG × vision/language vs decodability) | `decodability_rebuild/plot_native_grid.py` |
| Fig. 4 (within-family pairs) | `decodability_rebuild/plot_axis1_calibrated.py` |
| Fig. 7, 8 (all EEG × partner pairs) | `shift_tests/plot_grid_vision_net.py --metric {mknn,cka}` |
| Fig. 9 (fMRI × vision/language) | `shift_tests_fmri/plot_fmri_scaling_null.py` |

## Setup

```bash
bash setup.sh && source venv/bin/activate   # Python deps
export HF_TOKEN=hf_...                      # or put it in tokens/hf_token.txt
```

The EEG extractors also need the model repos
([BioFoundation](https://github.com/pulp-bio/BioFoundation), [NeuroLM](https://github.com/935963004/NeuroLM),
[STEEGFormer](https://github.com/LiuyinYang1101/STEEGFormer)), passed with `--<model>-repo`.
fMRI is read from `data/fmri/` (`extract_fmri.py --out-dir data/fmri`).
