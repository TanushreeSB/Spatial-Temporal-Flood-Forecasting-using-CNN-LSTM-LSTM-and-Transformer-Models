# Spatial-Temporal-Flood-Forecasting-using-CNN-LSTM-LSTM-and-Transformer-Models

Flood forecasting is vital for reducing the impact of natural disasters on communities. Traditional statistical models often fail to capture the complex spatial and temporal dependencies of rainfall and river flow.
This research proposes a deep learning–based system using LSTM, CNN-LSTM, and Transformer models to predict flood severity based on past hydrological time-series data.
By comparing these models, we assess how temporal memory (LSTM), spatial feature extraction (CNN-LSTM), and attention mechanisms (Transformer) influence prediction accuracy.

## Methodology (Steps)

- Data Collection: Rainfall, river discharge, and water level data from open hydrology datasets (e.g., Kaggle / IMD / CWC).

- Data Preprocessing: Handle missing data, normalization, and convert to sequences (sliding window).

- Model Development:

LSTM for temporal sequence learning.

CNN-LSTM for spatial-temporal feature fusion.

Transformer for attention-based sequence forecasting.

- Evaluation Metrics: RMSE, MAE, R² score.

## Visualization:

Predicted vs Actual Flood Level

Loss curves

Comparative performance bar graph
