# ☢ Nuclear Auxiliary Cooling Pump Monitoring System

> **LSTM-based predictive maintenance dashboard** — trained on NASA CMAPSS turbofan degradation data and deployed as a retro-nuclear instrument interface simulating fleet-wide health monitoring of auxiliary cooling pumps in a nuclear power facility.

![Python](https://img.shields.io/badge/Python-3.9+-1e7a06?style=flat-square&labelColor=050a03&color=39ff14)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-1e7a06?style=flat-square&labelColor=050a03&color=39ff14)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-1e7a06?style=flat-square&labelColor=050a03&color=39ff14)
![License](https://img.shields.io/badge/License-MIT-1e7a06?style=flat-square&labelColor=050a03&color=39ff14)
![Status](https://img.shields.io/badge/Status-Simulation-ffb000?style=flat-square&labelColor=050a03&color=ffb000)

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Model Pipeline](#model-pipeline)
- [Dataset](#dataset)
- [Dashboard Interface](#dashboard-interface)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Technical Deep Dive](#technical-deep-dive)
- [Design Philosophy](#design-philosophy)
- [Limitations & Future Work](#limitations--future-work)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project implements a full end-to-end **predictive maintenance pipeline** using a deep learning approach, deployed through an interactive monitoring dashboard. The system is designed to answer a single operational question:

> *How many operational cycles does each pump have remaining before it requires maintenance?*

The model ingests multivariate time-series sensor readings, learns to reconstruct normal operational behaviour using an **LSTM Autoencoder**, and flags anomalies when reconstruction error exceeds a per-unit learned threshold. The remaining useful life (RUL) estimate — referred to as **lead time** throughout this project — represents the number of operational cycles between the first confirmed anomaly detection and end-of-life, giving maintenance teams actionable advance warning.

The system is framed as a simulation of nuclear auxiliary cooling pump (NACP) monitoring, demonstrating how a data-driven predictive maintenance model trained on publicly available industrial degradation data could be adapted to domain-specific infrastructure monitoring.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│                                                             │
│   NASA CMAPSS          Sensor          Sequence             │
│   FD001 Dataset  ───►  Scaling   ───►  Builder              │
│   (21 sensors)         (per-unit)      (sliding window)     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    MODEL LAYER                              │
│                                                             │
│   LSTM Encoder  ───►  Bottleneck  ───►  LSTM Decoder        │
│   (compression)       (latent)         (reconstruction)     │
│                                                             │
│   Reconstruction Error  ───►  Per-Engine Threshold          │
│   (MSE)                        Comparison                   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    DETECTION LAYER                          │
│                                                             │
│   Anomaly        Consecutive        Lead Time               │
│   Flags     ───►  Logic       ───►  Computation             │
│   (binary)       (debounce)         (cycles to failure)     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    DASHBOARD LAYER                          │
│                                                             │
│   Streamlit App  ───►  Fleet Summary  ───►  Pump Drilldown  │
│   (app.py)             KPI Panels          Sensor Trends    │
│                        Status Alerts       Data Table       │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Pipeline

### 1. Data Loading & Preprocessing

Raw sensor readings are loaded via `src/data_loader.py`. Each sensor channel is independently scaled using a **per-sensor `StandardScaler`**, fitted on the training split and persisted to `models/scalers.pkl`. This preserves the relative magnitude relationships between engines while normalising the input space.

### 2. Sequence Construction

`src/sequence_builder.py` implements a **sliding window** approach. For each engine, overlapping sequences of fixed length are extracted from the time-series, producing tensors of shape `(n_sequences, seq_length, n_sensors)`. Sequences respect engine boundaries — no window spans two engines.

### 3. LSTM Autoencoder

The core model (`models/lstm_autoencoder.keras`) is an encoder-decoder architecture:

- **Encoder**: One or more LSTM layers that compress the input sequence into a fixed-size latent representation
- **Bottleneck**: The compressed latent state capturing normal operational patterns
- **Decoder**: LSTM layers that reconstruct the original sequence from the latent state

The model is trained exclusively on **healthy operating data** (early-life cycles). It learns to reconstruct normal sensor patterns with low error. When a unit begins to degrade, its sensor signatures deviate from the learned normal distribution, and reconstruction error rises.

### 4. Reconstruction Error & Thresholding

`src/model_utils.py` computes **Mean Squared Error (MSE)** between the input sequence and its reconstruction for each window. Each engine receives its own anomaly threshold computed from its training error distribution (stored in `models/engine_thresholds.pkl`), making the detection robust to natural inter-unit variability.

### 5. Consecutive Anomaly Logic

`src/evaluation.py:apply_consecutive_logic()` implements a **debounce filter**: a unit is only flagged as anomalous when a minimum number of consecutive windows exceed the threshold. This suppresses transient spikes and ensures the detection signal is stable before triggering a lead time calculation.

### 6. Lead Time Computation

`src/evaluation.py:compute_lead_times()` calculates the number of cycles between the **first confirmed anomaly** and the **last observed cycle** for each engine. This is the core operational output — the advance warning window available to maintenance teams before unit failure.

---

## Dataset

This project uses the **NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** dataset, specifically the **FD001** subset.

| Property | Value |
|---|---|
| Source | NASA Ames Prognostics Data Repository |
| Subset | FD001 |
| Operating Conditions | Single (Sea Level) |
| Fault Modes | Single (HPC Degradation) |
| Training Units | 100 engines |
| Test Units | 100 engines |
| Sensors | 21 channels |
| Features used | Subset of informative sensors (config-defined) |

> **Domain transfer note:** The CMAPSS dataset captures turbofan engine degradation. This project treats each engine as an analogue for an auxiliary cooling pump unit, demonstrating how a degradation model trained on publicly available industrial data can be evaluated in the context of nuclear facility maintenance monitoring. The simulation framing is explicit throughout the interface.

**Reference:**
> Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* International Conference on Prognostics and Health Management (PHM08).

---

## Dashboard Interface

The monitoring dashboard is built with **Streamlit** and styled with a comprehensive custom CSS layer implementing a **retro-nuclear Pip-Boy aesthetic** — monochromatic phosphor green on black, VT323 and Share Tech Mono typography, CRT scanline overlays, and DEFCON-style status panels.

### Interface Sections

**Page Masthead**
Full-width facility nameplate with suite version, unit identifier, and a 7-line boot sequence panel displaying model configuration on load.

**Fleet Status Broadcast**
A full-width dynamic status panel that evaluates fleet-wide lead time distribution and renders one of three states:
- `● SYS — ALL SYSTEMS NOMINAL` (green) — no units below 50 cycle threshold
- `⚠ CAU — CAUTION ADVISORY` (amber) — 1–2 units approaching threshold
- `⚠ ALT — CRITICAL ALERT` (red, pulsing) — 3+ units below threshold

**Fleet Health Summary (SEC-01)**
Three KPI readout panels displaying total monitored pumps, fleet average lead time, and low-lead-time unit count. Each panel is colour-coded by severity and contains a live status badge with a blinking indicator dot.

**Per-Pump Lead Time (SEC-02)**
A bar chart showing lead time for every pump in the fleet. Bars are individually colour-coded (green/amber/red) by threshold. Dual reference lines show fleet average (amber dashed) and the 50-cycle warning threshold (red dotted).

**Raw Telemetry Register**
A scrollable data table (10 rows visible, all 100 accessible via scroll) showing Pump ID, Lead Time, and Status for every unit. Rows are colour-coded by severity with a frozen header column.

**Individual Pump Inspection (SEC-03)**
Operator-selectable drilldown into any individual pump. Includes a per-pump status broadcast, a sensor trend line chart with phosphor glow fill, and two operator control selectors for pump ID and sensor channel.

**System Footer**
Facility identification, model provenance, configuration summary, and simulation disclaimer.

---

## Project Structure

```
predictive-maintenance-nuclear/
│
├── app.py                        # Streamlit dashboard (single entry point)
│
├── models/
│   ├── lstm_autoencoder.keras    # Trained LSTM Autoencoder weights
│   ├── scalers.pkl               # Per-sensor StandardScalers
│   ├── engine_thresholds.pkl     # Per-engine anomaly thresholds
│   └── model_config.json         # Selected sensors, sequence length, etc.
│
├── src/
│   ├── data_loader.py            # Raw data ingestion & preprocessing
│   ├── sequence_builder.py       # Sliding window sequence construction
│   ├── model_utils.py            # Reconstruction error computation
│   └── evaluation.py             # Consecutive logic & lead time computation
│
├── notebooks/                    # Training & experimentation notebooks
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Installation

### Prerequisites

- Python 3.9+
- `pip` and `venv` (recommended)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/predictive-maintenance-nuclear.git
cd predictive-maintenance-nuclear
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Dashboard web framework |
| `tensorflow` | LSTM Autoencoder model |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Chart rendering |
| `scikit-learn` | Sensor scaling (`StandardScaler`) |
| `joblib` | Model artefact serialisation |

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`.

> **Note:** All model artefacts (`lstm_autoencoder.keras`, `scalers.pkl`, `engine_thresholds.pkl`, `model_config.json`) must be present in the `models/` directory before launching. These are generated by running the training notebooks.

---

## Technical Deep Dive

### Why an Autoencoder for Anomaly Detection?

Traditional supervised RUL prediction requires labelled end-of-life data for every unit — often impractical in production environments where failure data is scarce. An **unsupervised autoencoder** approach sidesteps this requirement:

1. Train only on **healthy data** (no failure labels needed)
2. The model learns the manifold of normal behaviour
3. Anomalies are detected as deviations from this manifold
4. No assumption is made about failure mode or degradation trajectory

This makes the approach inherently more generalisable to new units and failure modes.

### Per-Engine Adaptive Thresholds

A single global threshold fails in practice because units exhibit natural variability in their baseline reconstruction error. A pump running at slightly different steady-state conditions will have a higher baseline error even when healthy.

The solution is to compute a threshold **per engine** from that engine's own error distribution during its healthy phase:

```
threshold_i = mean(error_i_healthy) + k * std(error_i_healthy)
```

Where `k` is a sensitivity hyperparameter tuned during validation. This approach ensures that detection is sensitive to *relative change* within a unit rather than *absolute error level* across units.

### Consecutive Logic (Debouncing)

Raw reconstruction error is noisy. A single anomalous window may result from a transient sensor spike, not genuine degradation onset. The consecutive logic filter requires `n` consecutive windows to exceed the threshold before flagging a unit, dramatically reducing false positive rate at a small cost in detection latency.

### Lead Time as an Operational Metric

Unlike a raw RUL estimate (which requires knowing the failure point in advance), lead time is computed entirely from **observed data**:

```
lead_time_i = final_cycle_i - first_confirmed_anomaly_cycle_i
```

This is a retrospective metric on the test set — it tells us how much advance warning the system *would have provided* if deployed at the start of the unit's operational life. On the training set it validates detection sensitivity; on the test set it validates operational utility.

---

## Design Philosophy

### Pip-Boy Nuclear Aesthetic

The dashboard interface draws from the **Pip-Boy** design language of the Fallout game series — a retro-futuristic computer terminal aesthetic that maps intuitively onto nuclear facility monitoring:

- **Monochromatic phosphor green** (`#39ff14`) on near-black (`#050a03`) — mimics a CRT phosphor screen
- **CRT scanline overlay** — a repeating CSS gradient at `z-index: 9999` creates authentic screen texture
- **Phosphor sweep line** — a single bright horizontal line descends the viewport every 14 seconds, simulating a CRT refresh cycle
- **VT323 display font** — a bitmap typeface designed to replicate dot-matrix terminal output
- **Share Tech Mono body font** — a monospace font evoking military/industrial documentation
- **DEFCON-style status panels** — alerts render as system broadcasts with severity-coded borders and pulsing animations
- **Boot sequence** — a 7-line terminal readout displaying live model configuration on every page load

The aesthetic serves a functional purpose: it gives the interface a clear identity that signals the domain (industrial monitoring), creates visual hierarchy through the green/amber/red severity system, and makes the dashboard genuinely memorable as a portfolio piece.

### Implementation Constraint

The entire aesthetic overhaul is contained within a **single file** (`app.py`) using Streamlit's `st.markdown(unsafe_allow_html=True)` for CSS injection and HTML component rendering. No additional dependencies, no separate CSS files, no JavaScript. This constraint was chosen deliberately to demonstrate what can be achieved within Streamlit's standard tooling.

---

## Limitations & Future Work

### Current Limitations

**Simulation framing** — The system is trained on turbofan engine data (NASA CMAPSS), not actual nuclear cooling pump sensor data. The domain transfer is conceptual. A production deployment would require retraining on facility-specific sensor data with appropriate validation protocols.

**Retrospective lead time** — Lead time is computed post-hoc from test set data. A true deployment would need to estimate remaining useful life in real-time, which requires a different output formulation (regression rather than anomaly detection).

**No real-time data ingestion** — The current implementation loads a static processed dataset. A production system would interface with a SCADA/DCS data stream.

**Single fault mode** — FD001 simulates a single fault type under a single operating condition. Real pumps exhibit multiple degradation modes simultaneously.

### Potential Extensions

- **Real-time streaming** — Connect to a live data source via Kafka or MQTT and update predictions on a rolling window
- **RUL regression head** — Add a supervised regression output alongside the autoencoder to provide a quantitative RUL estimate rather than a binary anomaly flag
- **Multi-fault detection** — Train on FD002/FD003/FD004 subsets (multiple operating conditions, multiple fault modes) and add fault classification
- **Uncertainty quantification** — Implement Monte Carlo Dropout or Bayesian LSTM to provide confidence intervals on lead time estimates
- **Automated alerting** — Integrate outbound webhook notifications when units breach thresholds
- **Historical trend logging** — Persist lead time history to a database and render degradation trend curves alongside sensor readings

---

## Acknowledgements

- **NASA Ames Prognostics Center of Excellence** — for releasing the CMAPSS dataset publicly, enabling reproducible research in prognostics and health management
- **Saxena et al. (2008)** — for the original dataset publication and damage propagation modelling framework
- **Bethesda Game Studios** — for the Pip-Boy design language that inspired the dashboard aesthetic
- **Streamlit** — for providing a rapid prototyping framework that made single-file dashboard deployment feasible

---

> *This system is a simulation. It is not certified for operational use in any nuclear or safety-critical facility. All monitoring outputs are for demonstration and research purposes only.*

---

<div align="center">

**[ NACP-7 // VAULT-TEC INDUSTRIAL MONITORING SUITE // REV 4.7.1 ]**

*Unit 7 &nbsp;|&nbsp; Primary Loop &nbsp;|&nbsp; Sector D &nbsp;|&nbsp; Simulation Mode*

</div>
