# transferattacktInterns

A clean intern-facing repository for a **vanilla transfer-attack** exercise on face verification.

This repo is intentionally a standalone vanilla transfer-attack exercise.

## What is included
- a vanilla-only attack core with 5 implemented baseline attacks
- a small curated subset for quick experiments
- existing baseline results on that subset
- a Track A assignment note for implementing a **new** transfer attack from recent literature

## Implemented baseline attacks
- PGD
- MI-FGSM
- TI-FGSM
- SI-NI-FGSM
- MI-ADMIX-DI-TI

## Completed student-contributed attacks
- `BPA_CNN` — Om Singh Rawat, IIT Delhi — *Rethinking the Backward Propagation for Adversarial Transferability* (NeurIPS 2023)
- `PGN` — Chidroopa Kanaparthy, Maharaja Agrasen Institute of Technology (MAIT) — *Boosting Adversarial Transferability by Achieving Flat Local Maxima* (NeurIPS 2023)
- `MIG` — Lakshita Sharma, Bhagwan Parshuram Institute of Technology — *Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients* (ICCV 2023)
- `ADAMSI_FGM` — Bhumika Singh, SRM Institute of Science and Technology — *On the Convergence of an Adaptive Momentum Method for Adversarial Attacks* (AAAI 2024)
- `BSR` — Chirag Sharma, IIIT Vadodara — *Boosting Adversarial Transferability by Block Shuffle and Rotation* (CVPR 2024)
- `DECOWA` — Om Singh Rawat, IIT Delhi — *Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping* (AAAI 2024)
- `SIA_MI_TI` — Janhavi Kishor, SRM University — *Structure Invariant Transformation for better Adversarial Transferability* (ICCV 2023), adapted with MI-FGSM and TI-FGSM
- `SIA` — Aditi Raj, IGDTUW — *Structure Invariant Transformation for better Adversarial Transferability* (ICCV 2023)
- `OPS` — Kkartik Aggarwal, Delhi Technological University (DTU) — *Boosting Adversarial Transferability through Augmentation in Hypothesis Space* (CVPR 2025)
- `ATT_CNN` — Keshav Raj, IIIT Delhi — CNN-side ATT-inspired adaptation motivated by *ATT* (NeurIPS 2024)
- `ATT_CNN_PATCH` — Pratyush Kumar, KCC Institute of Technology and Management (A.K.T.U.) — CNN-side ATT-inspired stochastic patch-masking adaptation motivated by *ATT* (NeurIPS 2024)
- `LI_BOOST_MI` — Charushi, IGDTUW — MI-style logarithmic-shift boosting transfer attack
- `GRA` — Krish Bansal, Delhi Technological University (DTU) — *Boosting Adversarial Transferability via Gradient Relevance Attack* (ICCV 2023)
- `IDAA` — Arnav Asthana, IIIT Delhi — student-contributed identity-aware / diversified attack adaptation integrated into the shared CNN face-verification pipeline
- `DPA_HMA` — Kushal Khemka, Delhi Technological University (DTU) — *Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation* (CVPR 2025)
- `DYNAMIC_MORPH` — Puneet Kumar, Indian Institute of Information Technology, Senapati, Manipur — student-submitted morph-style semantic face-region mixing adaptation, presented as D-FMA

## Current official subset baseline
- SI_NI_FGSM: 29.17%
- MI_FGSM: 26.67%
- MI_ADMIX_DI_TI: 24.17%
- TI_FGSM: 20.42%
- PGD: 16.67%

## Current experiment setup
### Attacker (surrogate) models
- Facenet512
- ArcFace
- GhostFaceNet
- VGG-Face

### Victim models
- Facenet512
- ArcFace
- GhostFaceNet
- VGG-Face
- IR152

Victim evaluation excludes self-transfer pairs.

## Small working subset
See:
- `docs/subset_input_pairs.csv`
- `results_baseline/subset_raw_similarities_long.csv`
- `results_baseline/subset_attack_summary.csv`
- `results_baseline/subset_attack_summary_by_goal.csv`
- `results_baseline/subset_attacker_victim_summary.csv`

## Baseline reproducibility note
- The baseline CSV files included in `results_baseline/` are the official reference for this repo.
- These baseline summaries were prepared from a precomputed raw-similarity source and are the values students should use for comparison.
- If you rerun the attack generation pipeline locally, you may observe small differences in breach rate and impact because adversarial sample generation is not fully deterministic across runs and environments.
- In particular, clean similarities are expected to stay essentially unchanged, while adversarial similarities may vary slightly.

## Goal for interns
Implement one new transfer attack that is **not already present in this repo**, adapt it to the face-verification setting, and compare it against the 5 vanilla baselines using breach rate and impact on the provided subset.

## Upstream attack source to inspect
- https://github.com/Trustworthy-AI-Group/TransferAttack

## Recommended starting point
Read:
- `docs/trackA_assignment.md`
- `core/README.md`
- `results_baseline/baseline_notes.md`
- `results_student_attacks/bsr/README.md`
- `results_student_attacks/bsr/bsr_vs_current_baseline_summary.csv`
- `results_student_attacks/decowa/README.md`
- `results_student_attacks/decowa/decowa_vs_current_baseline_summary.csv`
- `results_student_attacks/sia/README.md`
- `results_student_attacks/sia_mi_ti/README.md`
- `results_student_attacks/sia_mi_ti/sia_mi_ti_vs_current_baseline_summary.csv`
- `results_student_attacks/bpa_cnn/README.md`
- `results_student_attacks/bpa_cnn/bpa_cnn_vs_current_baseline_summary.csv`
- `results_student_attacks/ops/README.md`
- `results_student_attacks/pgn/README.md`
- `results_student_attacks/adamsi_fgm/README.md`
- `results_student_attacks/att_cnn/README.md`
- `results_student_attacks/att_cnn/att_cnn_vs_current_baseline_summary.csv`
- `results_student_attacks/li_boost_mi/README.md`
- `results_student_attacks/li_boost_mi/li_boost_mi_vs_current_baseline_summary.csv`
- `results_student_attacks/mig/README.md`
- `results_student_attacks/mig/mig_vs_baselines_summary.csv`
- `results_student_attacks/gra/README.md`
- `results_student_attacks/idaa/README.md`
- `results_student_attacks/dpa_hma/README.md`
- `results_student_attacks/dpa_hma/dpa_hma_vs_current_baseline_summary.csv`
- `results_student_attacks/dynamic_morph/README.md`
