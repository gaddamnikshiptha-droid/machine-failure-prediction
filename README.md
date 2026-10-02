# Machine Failure Prediction

Predicts the Remaining Useful Life (RUL) of turbofan engines, that is, how many operating cycles remain before failure, from 14 sensor streams in the NASA C-MAPSS FD001 dataset.

## Approach

1. **Data preparation.** Seven constant sensor channels are removed, the remaining fourteen are min-max scaled, and each engine's history is cut into sliding 30-cycle windows. The RUL target is capped at 125 cycles, the standard piecewise-linear formulation for this benchmark. Validation engines are held out by unit to prevent leakage.
2. **SVR baseline.** Each window is compressed into mean, last value and linear trend per sensor, and an RBF-kernel Support Vector Regressor is trained on these features.
3. **LSTM model.** Two stacked LSTM layers learn degradation patterns directly from the raw sequence. The recurrence is unrolled so the model converts cleanly to TensorFlow Lite.
4. **Edge deployment.** The LSTM is converted to TensorFlow Lite in float32 and INT8 dynamic-range variants and benchmarked on size, latency and accuracy.
5. **Dashboard.** A Streamlit application shows fleet health, per-engine RUL trajectories, sensor trends and model comparisons.

Evaluation uses RMSE, MAE and the NASA asymmetric scoring function, which penalises late failure predictions more heavily than early ones. Full results are in [RESULTS.md](RESULTS.md).

## Project Structure

| File | Purpose |
|---|---|
| `cmapss.py` | Data loading, scaling, windowing, features and metrics |
| `train.py` | Trains the SVR baseline and the LSTM, saves models and metrics |
| `export_tflite.py` | TensorFlow Lite conversion, quantization and benchmarking |
| `app.py` | Streamlit dashboard |

## Running

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place train_FD001.txt, test_FD001.txt and RUL_FD001.txt in data/
python train.py
python export_tflite.py
streamlit run app.py
```

## Dataset

A. Saxena and K. Goebel, "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository.
