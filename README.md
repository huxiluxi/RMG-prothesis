# RMGrealtime

Real-time Radio Myography (RMG) gesture recognition framework using a LiteVNA, RF signal processing, and machine learning.

This project acquires calibrated S-parameter measurements from a LiteVNA device and performs real-time hand gesture classification using RF sensing and machine learning pipelines.

---

## Features

- Real-time RF gesture recognition
- LiteVNA integration for high-speed acquisition
- SOLT calibration workflow
- S-parameter conversion and calibration pipeline
- Feature extraction from RF measurements
- Multiple ML classifier backends
- Live RF sweep visualization
- Dataset collection and analysis tools
- PCA/LDA dimensionality reduction support
- Cross-run evaluation and model comparison
- Optional robotic hand integration support

---

## Repository Structure

```text
RMGrealtime/
├── acquisition/          # LiteVNA communication layer
├── control/              # Optional robotic hand control
├── core/                 # Shared config, logging, and types
├── decision/             # Decision filtering and commitment logic
├── documentation/        # Manuals and documentation PDFs
├── output/               # Generated plots and evaluation outputs
├── perception/           # ML pipeline and feature extraction
├── rf/                   # RF calibration and S-parameter handling
│
├── calibration_SOLT.py  # SOLT calibration workflow
├── collect_data.py      # Dataset acquisition script
├── live_view.py         # Real-time RF visualization
├── model_selection.py   # Model benchmarking and evaluation
├── realtimeRMG.py       # Main real-time inference application
└── README.md
```

---

## System Overview

The pipeline consists of:

1. **RF Acquisition**
   - LiteVNA captures RF sweeps over a configured frequency range.

2. **S-Parameter Processing**
   - Raw sweeps are converted into calibrated S-parameters.

3. **Feature Extraction**
   - Magnitude-based features are extracted from the RF response.

4. **Machine Learning Inference**
   - Features are passed through trained classifiers for gesture prediction.

5. **Decision Filtering**
   - Temporal commitment filtering stabilizes predictions.

---

## Supported Gestures

The default gesture set includes:

- PalmarPinch
- 2FingerPinch
- LateralPinch
- Wrap
- Open
- Relaxed

---

## Requirements

### Hardware

- LiteVNA
- RF sensing setup / electrodes / antenna system
- USB serial connection
- Optional robotic hand hardware

### Software

Recommended:

- Python 3.10+
- Windows or Linux

---

## Python Dependencies

Install required packages:

```bash
pip install numpy pandas matplotlib scikit-learn pyserial scipy
```

Additional packages may be required depending on your environment and hardware integration.

---

## Configuration

Project configuration is stored in:

```text
core/config.py
```

Typical configuration parameters include:

- COM port
- Frequency range
- Number of sweep points
- Calibration file path
- Hand control configuration
- CAN interface settings

---

## Calibration

Perform SOLT calibration before running inference.

### Run calibration

```bash
python calibration_SOLT.py
```

The script supports:

- Loading an existing calibration
- Creating a new SOLT calibration

Calibration files are stored in:

```text
rf/
```

---

## Live RF Visualization

Visualize raw and calibrated S-parameters in real time:

```bash
python live_view.py
```

Features:

- Real-time sweep updates
- Raw vs calibrated comparison
- S11 and S21 plotting

---

## Data Collection

Collect gesture datasets for training and evaluation:

```bash
python collect_data.py
```

The script:

- Guides the user through gesture acquisition
- Performs countdown-based recording
- Saves all measurements into CSV datasets

---

## Model Training and Evaluation

Use the model selection script to benchmark classifiers and dimensionality reduction methods.

### Run evaluation

```bash
python model_selection.py
```

Supported classifiers include:

- k-NN
- Centroid
- Gaussian Mixture Model (GMM)
- Mahalanobis
- LDA
- SVM

Features:

- Adaptive PCA evaluation
- Cross-run testing
- Confusion matrix generation
- Accuracy benchmarking

Generated outputs are saved in:

```text
output/
```

---

## Real-Time Gesture Recognition

Run the complete inference pipeline:

```bash
python realtimeRMG.py
```

Pipeline stages:

1. LiteVNA acquisition
2. Calibration correction
3. Feature extraction
4. Classification
5. Decision filtering
6. Gesture output

Optional robotic hand control hooks are included but currently commented out.

---

## Machine Learning Pipeline

Main ML modules:

| Module | Description |
|---|---|
| `perception/features.py` | RF feature extraction |
| `perception/classifier.py` | Classification models |
| `perception/trainer.py` | Training utilities |
| `decision/commitment.py` | Temporal smoothing/filtering |

---

## RF Processing

RF processing components:

| Module | Description |
|---|---|
| `rf/sparameters.py` | Sweep to S-parameter conversion |
| `rf/calibration.py` | Calibration handling |
| `acquisition/litevna.py` | LiteVNA communication |

---

## Example Workflow

### 1. Perform calibration

```bash
python calibration_SOLT.py
```

### 2. Verify live RF signals

```bash
python live_view.py
```

### 3. Collect gesture data

```bash
python collect_data.py
```

### 4. Train/evaluate models

```bash
python model_selection.py
```

### 5. Run real-time inference

```bash
python realtimeRMG.py
```

---

## Outputs

The `output/` directory contains generated evaluation artifacts such as:

- Confusion matrices
- Classification reports
- PCA analysis plots
- Accuracy comparison plots
- Calibration error visualizations

---

## Documentation

Included manuals:

- `documentation/07 Product Manual.pdf`
- `documentation/LiteVNA_User-Guide.pdf`

---

## Authors

Jens-Ulrik Ladekjær-Mikkelsen
Masters student

---
