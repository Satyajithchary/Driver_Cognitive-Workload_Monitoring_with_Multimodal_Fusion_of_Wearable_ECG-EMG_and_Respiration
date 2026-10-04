# Driver Cognitive-Workload Monitoring with Multimodal Fusion of Wearable ECG, EMG and Respiration

Code for subject-independent recognition of cognitive workload from three wearable sensors on the public ADABase dataset. The sensors are 2-lead ECG, trapezius EMG and chest respiration. Each sensor stream is encoded by a 1D transformer, and nine fusion strategies are compared under leave-one-subject-out (LOSO) evaluation with three subject-calibration protocols.

The accompanying manuscript is under review at IEEE Sensors Letters.

## Contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Method](#method)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Highlights

- Cross-attention fusion reaches a macro-F1 of 0.781 and an AUROC of 0.841 for driving level 1 vs. 3 across 29 unseen subjects.
- The informative sensor depends on the task. Trapezius EMG dominates driving, whereas ECG dominates the n-back task.
- The gain of fusion over the best single sensor is small and not significant after Holm correction (n = 29).
- Calibration with unlabeled task data improves macro-F1 by 10 to 22 points and is significant for 10 of 12 models. A 5-min rest recording is not sufficient.

## Dataset

The code uses the ADABase cognitive-drive public release 0.1.0 (Oppelt et al., *Sensors* 23:340, 2023). The release contains 30 subjects, with one HDF5 file per subject (about 26 GB in total). The dataset is not redistributed here, so it has to be requested from the dataset authors. The path to the HDF5 files is set in `configs/adabase.yaml`.

| Task | Classes | Data per subject | Chance |
|---|---|---|---|
| `kdrive_bin` | driving level 1 vs. level 3 | 2 x 300 s | 0.50 |
| `kdrive_3` | driving level 1 / 2 / 3 | 3 x 300 s | 0.33 |
| `nback_bin` | low (stimulus-only, 1-back) vs. high (2-back, 3-back) | 4 x 120 s | 0.50 |
| `nback_3` | 1-back / 2-back / 3-back | 3 x 120 s | 0.33 |

The 5-min rest baseline of each study serves only as the calibration reference and is never scored. One subject per study was excluded by quality rules fixed before modelling, which leaves 29 subjects per task.

<p align="center">
  <img src="figures/eda/manipulation_check.png" width="900" alt="NASA-TLX and task performance per level">
</p>
<p align="center"><em>Manipulation check. NASA-TLX and task performance per level (Friedman test, n = 30).</em></p>

## Method

**Pre-processing.** Each sensor is filtered, quality-checked and resampled into its own stream:

- ECG (500 Hz) is band-pass filtered (0.5 to 40 Hz) and corrected for polarity, then R-peaks and a signal quality index are computed per lead. The ECG stream contains both leads plus the instantaneous heart rate at 125 Hz.
- EMG (1000 Hz) is band-pass filtered (20 to 450 Hz) and notch-filtered at 50 Hz. It is then converted to log-RMS envelopes in three sub-bands at 50 Hz.
- Respiration (250 Hz) is band-pass filtered (0.05 to 1 Hz) and represented by the waveform plus the breathing rate at 25 Hz.

All streams are cut into 20-s windows.

<p align="center">
  <img src="figures/eda/stream_examples.png" width="900" alt="Pre-processed streams">
</p>
<p align="center"><em>Pre-processed ECG, EMG envelopes and respiration of one subject at rest, in 1-back and in 3-back.</em></p>

**Encoders.** Each stream passes through a convolutional stem and a patch convolution, followed by four pre-norm transformer layers (d = 128, 4 heads). Each single-stream encoder has about 0.66 M parameters.

**Fusion strategies.**

| Family | Strategy |
|---|---|
| Early | Joint transformer on the tokens of all streams |
| Intermediate | Concatenation, attention-weighted fusion |
| Late | Learned decision-level weighting, LXMERT-style cross-attention |
| Tensor | Bilinear TFN, low-rank LMF, higher-order polynomial (HOT-3) |
| Hierarchical | Fusion at every encoder depth with attention over levels |

**Calibration protocols.**

| Protocol | Reference set of the test subject | Analogue |
|---|---|---|
| `strict` | none | plug-and-play |
| `enroll` (main) | 5-min rest baseline, mean subtraction | short rest recording before use |
| `calib` | all unlabeled task windows, full standardisation | offline upper bound |

**Evaluation.** Leave-one-subject-out over 29 folds, with 4 inner validation subjects for early stopping. Statistics treat the subject as the unit: Friedman test with Nemenyi critical difference, and Wilcoxon signed-rank tests with Holm correction.

## Repository structure

```
.
├── configs/
│   └── adabase.yaml          paths, tasks, protocols, window and training settings
├── fada/                     package: signal processing, data, models, training, statistics, plotting
│   └── models/fusions.py     the nine fusion strategies
├── scripts/
│   ├── a01_extract.py        chunked HDF5 extraction of ECG / RSP / EMG segments
│   ├── a02_preprocess.py     stream pre-processing, quality audit, exclusions
│   ├── a03_eda.py            cohort, manipulation check, physiology per level
│   ├── a06_train.py          LOSO training (sharded, resumable)
│   ├── run_training.sh       launches all main runs in parallel shards
│   ├── a07_analyze.py        per task / protocol tables, statistics, figures
│   ├── a08_ablations.py      stream pairs and window length
│   ├── a09_summary.py        cross-task summary and protocol tests
│   └── make_notebook.py      builds ADABase.ipynb
├── figures/                  generated figures
└── results/                  generated tables and predictions
```

## Installation

The code was tested with Python 3.12, PyTorch 2.11 and CUDA 12.8.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy scipy pandas tables neurokit2 scikit-learn statsmodels matplotlib pyyaml nbformat papermill
```

## Reproducing the results

The full pipeline can be run from the notebook:

```bash
python scripts/make_notebook.py
papermill ADABase.ipynb ADABase.ipynb --cwd .
```

The steps can also be run one by one:

```bash
python scripts/a01_extract.py 8                 # extraction with 8 workers
python scripts/a02_preprocess.py                # pre-processing and quality audit
python scripts/a03_eda.py                       # exploratory analysis
bash   scripts/run_training.sh                  # all LOSO runs (logs in logs/)
python scripts/a07_analyze.py --task kdrive_bin --protocol enroll
python scripts/a08_ablations.py --run
python scripts/a08_ablations.py --summarise
python scripts/a09_summary.py
```

A single model, task and protocol can be trained with:

```bash
python scripts/a06_train.py --task kdrive_bin --protocol enroll --models late_xattn --seeds 0
```

## Results

### Main results (enroll protocol)

The table reports window-level macro-F1 and AUROC under LOSO evaluation. The n-back binary values are means of 3 seeds (SD at most 0.033). All other columns use seed 0.

| Model | k-drive bin. F1 | k-drive bin. AUROC | k-drive 3-class F1 | n-back bin. F1 | n-back bin. AUROC | n-back 3-class F1 |
|---|---|---|---|---|---|---|
| ECG only | .544 | .570 | .379 | .605 | .652 | .391 |
| EMG only | .770 | .804 | .527 | .558 | .568 | .288 |
| RSP only | .586 | .631 | .406 | .547 | .586 | .354 |
| Early (joint tokens) | .710 | .797 | .451 | .560 | .599 | .363 |
| Intermediate concat. | .711 | .788 | **.541** | .573 | .615 | .365 |
| Intermediate attention | .774 | .835 | .513 | .615 | .651 | .326 |
| Late (decision) | .732 | .804 | .526 | .610 | .676 | **.420** |
| Cross-attention (LXMERT) | **.781** | **.841** | .497 | **.648** | **.693** | .370 |
| Bilinear (TFN) | .736 | .795 | .497 | .599 | .641 | .371 |
| Low-rank (LMF) | .728 | .802 | .501 | .592 | .632 | .384 |
| Higher-order (HOT-3) | .740 | .815 | .516 | .602 | .654 | .364 |
| Hierarchical | .739 | .791 | .508 | .593 | .646 | .408 |
| Chance | .500 | .500 | .333 | .500 | .500 | .333 |

<p align="center">
  <img src="figures/summary/tasks_models_heatmap.png" width="800" alt="Macro-F1 per task and model">
</p>
<p align="center"><em>Enroll protocol. Window macro-F1 per task and model.</em></p>

### Statistical comparison

<p align="center">
  <img src="figures/kdrive_bin/enroll/cd_diagram.png" width="430" alt="CD diagram k-drive binary">
  <img src="figures/nback_bin/enroll/cd_diagram.png" width="430" alt="CD diagram n-back binary">
</p>
<p align="center"><em>Critical-difference diagrams of subject-level macro-F1 for k-drive binary (left) and n-back binary (right). Models joined by a bar do not differ significantly (Nemenyi, CD = 3.09 ranks, n = 29).</em></p>

### Sensor reliance

The table reports the change in macro-F1 when one stream is set to zero at test time (enroll protocol).

| Fusion | k-drive −ECG | k-drive −EMG | k-drive −RSP | n-back −ECG | n-back −EMG | n-back −RSP |
|---|---|---|---|---|---|---|
| Early | +.014 | **−.151** | −.012 | −.005 | +.039 | +.021 |
| Intermediate concat. | +.003 | **−.134** | +.006 | −.021 | +.026 | +.015 |
| Intermediate attention | −.040 | **−.189** | +.001 | **−.038** | +.002 | −.002 |
| Late (decision) | +.049 | **−.139** | −.016 | **−.043** | −.016 | +.010 |
| Cross-attention | −.073 | **−.192** | −.013 | **−.103** | −.040 | −.018 |
| Bilinear (TFN) | −.025 | **−.136** | −.005 | −.016 | −.013 | +.016 |
| Low-rank (LMF) | −.199 | **−.367** | −.088 | −.056 | **−.100** | −.060 |
| Higher-order (HOT-3) | +.007 | **−.136** | −.001 | **−.032** | +.013 | +.011 |
| Hierarchical | −.023 | **−.119** | +.024 | −.015 | +.006 | +.010 |

<p align="center">
  <img src="figures/kdrive_bin/enroll/missing_modality.png" width="430" alt="Missing stream k-drive">
  <img src="figures/nback_bin/enroll/missing_modality.png" width="430" alt="Missing stream n-back">
</p>
<p align="center"><em>Macro-F1 with all streams and with one stream removed, for k-drive binary (left) and n-back binary (right).</em></p>

### Calibration protocols (n-back binary)

The table reports subject-level macro-F1 for seed 0. The p-values come from the Wilcoxon signed-rank test with Holm correction over 24 tests.

| Model | strict | enroll | calib | enroll vs. strict p (Holm) | calib vs. enroll p (Holm) |
|---|---|---|---|---|---|
| ECG only | .486 | .605 | .705 | 0.221 | **0.021** |
| EMG only | .535 | .508 | .628 | 1.000 | 0.352 |
| RSP only | .488 | .490 | .667 | 1.000 | **<0.001** |
| Early (joint tokens) | .558 | .546 | .736 | 1.000 | **0.001** |
| Intermediate concat. | .549 | .515 | .736 | 1.000 | **<0.001** |
| Intermediate attention | .561 | .576 | .736 | 1.000 | **0.007** |
| Late (decision) | .583 | .585 | .735 | 1.000 | **0.018** |
| Cross-attention (LXMERT) | .572 | .600 | .749 | 1.000 | **0.018** |
| Bilinear (TFN) | .579 | .568 | .678 | 1.000 | **0.018** |
| Low-rank (LMF) | .516 | .587 | .583 | 1.000 | 1.000 |
| Higher-order (HOT-3) | .517 | .555 | .723 | 1.000 | **<0.001** |
| Hierarchical | .540 | .572 | .730 | 1.000 | **0.009** |

<p align="center">
  <img src="figures/summary/protocols_main_task.png" width="800" alt="Calibration protocols">
</p>
<p align="center"><em>Window macro-F1 of all models under the strict, enroll and calib protocols.</em></p>

### Ablations (n-back binary, enroll, seed 0)

<p align="center">
  <img src="figures/ablations/ablations.png" width="800" alt="Ablations">
</p>
<p align="center"><em>Stream pairs vs. all three streams (left) and window length of 10, 20 and 30 s (right).</em></p>

### Limitations

- Task order is fixed in ADABase, so workload and time-on-task are partly confounded.
- The secondary tasks and the ablations use a single training seed.
- All recordings come from a driving simulator.

## Citation

If this code is useful for your work, please cite the paper (details will be updated after acceptance):

```bibtex
@article{adabase_fusion_2026,
  title   = {Intelligent Driver Cognitive-Workload Monitoring Using Multimodal Fusion of Wearable ECG, EMG, and Respiration Sensors},
  author  = {Author One and Author Two and Author Three},
  journal = {IEEE Sensors Letters},
  year    = {2026},
  note    = {under review}
}
```

The dataset should be cited as:

```bibtex
@article{oppelt2023adabase,
  title   = {ADABase: A Multimodal Dataset for Cognitive Load Estimation},
  author  = {Oppelt, Maximilian P. and others},
  journal = {Sensors},
  volume  = {23},
  number  = {1},
  pages   = {340},
  year    = {2023}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details. The ADABase data are subject to the license of the dataset authors.
