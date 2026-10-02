# Results

Dataset: NASA C-MAPSS FD001 (100 test engines). RUL capped at 125 cycles.

## Model Comparison

| Model | RMSE | MAE | NASA Score | Train Time (s) |
|---|---|---|---|---|
| SVR | 13.69 | 10.54 | 378.3 | 2.4 |
| LSTM | 13.57 | 10.55 | 275.8 | 61.9 |

## Edge Deployment Benchmark

| Variant | Size (KB) | Latency (ms / sample) | Test RMSE |
|---|---|---|---|
| Keras (float32) | 435.4 | 0.281 | 13.57 |
| TFLite (float32) | 291.6 | 0.115 | 13.57 |
| TFLite (INT8 weights) | 206.7 | 0.115 | 13.52 |
