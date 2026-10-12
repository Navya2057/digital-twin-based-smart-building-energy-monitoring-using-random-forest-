# Digital Twin-Based Smart Building Energy Monitoring Using Random Forest

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![Status](https://img.shields.io/badge/status-research%20prototype-lightgrey)

A research prototype that forecasts hourly building electricity use with a **Random Forest**, replays telemetry through a **digital-twin-style monitoring pipeline**, and flags unusual consumption using **residual-based alerts**. The project includes a leakage-aware evaluation protocol, an interactive Streamlit dashboard, and the code used for the accompanying research manuscript.

> **Scope note.** This is a *replay-driven prototype*, not a live deployed twin. It emulates the physical-to-digital data flow with recorded data and has no actuation or closed-loop control. Following the digital model / shadow / twin classification of Kritzinger et al. (2018), it is best described as a **digital shadow prototype**.

---

## Table of Contents
1. [Highlights](#highlights)
2. [System overview](#system-overview)
3. [Dataset](#dataset)
4. [Repository structure](#repository-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Evaluation methodology](#evaluation-methodology)
8. [Results](#results)
9. [Limitations](#limitations)
10. [Roadmap](#roadmap)
11. [Authors](#authors)
12. [Citation and acknowledgments](#citation-and-acknowledgments)
13. [License](#license)

---

## Highlights
- **Forecasting:** one-hour-ahead energy prediction using a tuned Random Forest on calendar, weather, and lag features.
- **Monitoring:** residual-based anomaly alerts with an adjustable sensitivity threshold.
- **Interactive dashboard:** streaming-style replay with actual vs. predicted energy, alerts, and what-if queries (Streamlit).
- **Rigorous evaluation:** chronological rolling-origin evaluation, naive baselines (persistence, seasonal naive), linear regression and gradient boosting baselines, and block-bootstrap confidence intervals.
- **Reproducible:** fixed seeds, pinned dependency ranges, and scripts that regenerate every figure and table.

## System overview

```
 Building / meters ──► Telemetry replay ──► Hourly aggregation + feature engineering
 (UCI house data)       (stand-in for IoT)      (calendar, lag-1h, lag-24h, rolling-24h)
                                                        │
                                  State buffer ─────────┤
                              (last 24 h measured)      ▼
                                              Random Forest forecaster ──► Residual monitor (alerts)
                                                        │                          │
                                                        └──────────► Streamlit dashboard + what-if
```

| Layer | Implemented | Not implemented |
|---|---|---|
| Data acquisition | Replay of recorded telemetry | Live IoT/MQTT ingestion |
| Model | Random Forest forecaster, baselines | Automatic retraining, drift detection |
| Monitoring | Residual threshold alerts | Fault diagnosis on real labelled faults |
| Interface | Streamlit dashboard, what-if queries | Actuation / closed-loop control |

## Dataset
The project uses the public **UCI Appliances Energy Prediction** dataset (Candanedo et al., 2017):
10-minute records from a single low-energy house over about 4.5 months, with indoor sensors and weather from a nearby airport station.

- **Target:** hourly electricity use of appliances + lights (kWh/h), computed from six 10-minute readings.
- **Inputs used:** hourly mean outdoor temperature and humidity, calendar features, and lagged energy.
- **After preprocessing:** 3,265 hourly samples (the first partial hour and a 24-hour lag warm-up are removed).
- **Access:** UCI Machine Learning Repository, DOI `10.24432/C5VC8G`. Please check the dataset page for license terms before redistributing the raw file.

> **Note on occupancy.** The dataset has no occupancy measurement. An early version of this prototype added a randomly generated `occupancy` column; it is **synthetic**, is not used in the main results, and an ablation shows it adds no meaningful accuracy.

## Repository structure
```
.
├── app.py                      # Streamlit dashboard (replay, alerts, what-if)
├── requirements.txt
├── data/                       # datasets and replay stream
├── notebooks/
│   └── eda_and_walkthrough.ipynb   # exploratory analysis and walkthrough
├── src/
│   ├── preprocess.py           # loading, aggregation, cleaning
│   ├── features.py             # calendar, lag and rolling features
│   ├── train.py                # model training, evaluation, model export
│   └── twin.py                 # twin logic: state buffer, forecasting, alerts
├── models/                     # trained model files (created by train.py)
└── paper/                      # (optional) manuscript, experiment and figure scripts
```

## Installation
Requires **Python 3.10+**.

```bash
git clone <your-repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Train the model**
```bash
python src/train.py
```
Trains the baseline and Random Forest models on a chronological split and writes the trained model to `models/`.

**2. Launch the dashboard**
```bash
streamlit run app.py
```
Open the local URL printed in the terminal. Use the controls to step through or rewind the replay, change the alert sensitivity, and run what-if queries.

**3. Explore the analysis**
```bash
jupyter notebook notebooks/eda_and_walkthrough.ipynb
```

**4. (Optional) Reproduce the paper experiments**
```bash
cd paper/code
python run_experiments.py --raw ../data/energydata_complete.csv --out ../results
python make_figures.py
python make_tables.py
```

## Evaluation methodology
- **Chronological, leakage-aware:** test data always follows training data. Rolling-mean features use only past values (shifted by one step).
- **Rolling-origin evaluation:** expanding training window, 5 consecutive test blocks of 333 hours (about 14 days each).
- **Baselines:** persistence (`t-1`), seasonal naive (`t-24`), linear regression, histogram gradient boosting.
- **Tuning:** Random Forest hyperparameters are searched only on each fold's training data with `TimeSeriesSplit`.
- **Metrics and uncertainty:** MAE, RMSE, R²; 95% moving-block bootstrap intervals (block length 24 h).
- **Explainability:** permutation feature importance.
- **Anomaly monitoring:** evaluated on *synthetic injected* anomalies; alert thresholds are calibrated on held-out residuals.

## Results
Pooled over 5 rolling-origin folds (1,665 test hours), mean [95% CI]:

| Model | MAE (kWh/h) | RMSE (kWh/h) | R² |
|---|---|---|---|
| Persistence (t-1) | 0.214 [0.187, 0.237] | 0.401 [0.353, 0.441] | 0.239 |
| Seasonal naive (t-24) | 0.291 [0.253, 0.326] | 0.531 [0.464, 0.587] | -0.333 |
| Linear Regression | 0.219 [0.198, 0.239] | 0.355 [0.312, 0.393] | 0.403 |
| HistGradientBoosting | 0.216 [0.195, 0.235] | 0.360 [0.322, 0.396] | 0.387 |
| **Random Forest** | **0.210 [0.190, 0.229]** | **0.348 [0.309, 0.383]** | **0.426** |

**Key findings**
- The Random Forest has the best point estimates, but its MAE advantage over persistence and gradient boosting is **not statistically established** (paired bootstrap intervals include zero).
- Forecasts rely mainly on recent measured energy and time of day; outdoor weather contributes little.
- A synthetic occupancy proxy adds no meaningful accuracy.
- Calibrating the alert threshold on training residuals produces more false alarms than calibrating on held-out residuals.

Results come from a single house over about 4.5 months and should not be generalized to other buildings.

## Limitations
- Single house, short period (no full seasonal cycle); the target excludes heating/cooling loads.
- Observed (not forecast) weather is used as input, which is optimistic for real deployment.
- Random Forests cannot extrapolate beyond the training range (outdoor temperature in the data does not exceed about 26 °C), so what-if scenarios outside it are unreliable.
- Anomaly experiments use injected synthetic anomalies, not real labelled faults.
- Real-time synchronization, concept drift, and retraining policy are not studied.

## Roadmap
- [ ] Multi-building evaluation (e.g., ASHRAE Great Energy Predictor III)
- [ ] Live data ingestion (e.g., MQTT) with measured end-to-end latency
- [ ] Drift monitoring and retraining policies
- [ ] Tuned gradient boosting, deep-learning and probabilistic baselines
- [ ] SHAP-based explainability
- [ ] Evaluation on real labelled faults

## Author
- **Navya Vashistha

IILM University, Greater Noida, India.

## Citation and acknowledgments
If you use this work or the dataset, please cite:

- L. M. Candanedo, V. Feldheim, and D. Deramaix, "Data driven prediction models of energy use of appliances in a low-energy house," *Energy and Buildings*, vol. 140, pp. 81–97, 2017. doi: 10.1016/j.enbuild.2017.01.083
- W. Kritzinger et al., "Digital twin in manufacturing: A categorical literature review and classification," *IFAC-PapersOnLine*, vol. 51, no. 11, pp. 1016–1022, 2018. doi: 10.1016/j.ifacol.2018.08.474

Parts of the documentation, code and manuscript were drafted with the assistance of an AI language model (Claude, Anthropic); the authors reviewed and take responsibility for the content.

## License
Add a `LICENSE` file (for example MIT) to the repository. The UCI dataset is subject to its own license terms; check the dataset page before redistributing it.
