# Spatial-Temporal Flood Forecasting: CNN-LSTM vs ConvLSTM vs CNN-Transformer

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random
import os


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# -----------------------
# Synthetic dataset generator
def generate_synthetic_spatiotemporal_data(n_samples=2000, seq_len=10, H=64, W=64):
    """
    Creates synthetic sequences of 'rainfall maps' (seq_len x H x W x 1)
    and a scalar target (flood level) that depends on spatial pattern intensity
    and temporal accumulation.
    """
    X = np.zeros((n_samples, seq_len, H, W, 1), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        base = np.zeros((H, W), dtype=np.float32)
        # number of rain cells
        k = np.random.randint(1, 5)
        for _ in range(k):
            cx = np.random.randint(10, H-10)
            cy = np.random.randint(10, W-10)
            sigma = np.random.uniform(3, 10)
            amp = np.random.uniform(0.4, 1.0)
            # gaussian blob
            xv, yv = np.meshgrid(np.arange(W), np.arange(H))
            blob = amp * np.exp(-((xv-cx)**2 + (yv-cy)**2)/(2*sigma**2))
            base += blob

        # normalize base
        base = base / (base.max() + 1e-6)

        # create sequence by shifting and adding noise
        seq = []
        accum = 0.0
        for t in range(seq_len):
            # small random shift
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            shifted = np.roll(base, shift=(shift_x, shift_y), axis=(0,1))
            noise = np.random.normal(scale=0.03, size=(H,W))
            frame = np.clip(shifted + noise, 0, 1)
            seq.append(frame[..., None])
            accum += frame.sum() * (1 + 0.1 * np.random.randn())

        # the target flood level is noisy function of accum + random catchment factor
        catchment_factor = np.random.uniform(0.8, 1.4)
        y_val = catchment_factor * accum / (H*W*seq_len)  # normalize scale
        y_val = 2.0 * np.tanh(1.2 * y_val) + 0.05 * np.random.randn()

        X[i] = np.stack(seq, axis=0)
        y[i] = y_val

    # rescale X to [0,1] already; scale y to zero-mean unit-std for stable training
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / (y_std + 1e-9)
    return X, y_norm, y_mean, y_std

SEQ_LEN = 10
H, W = 64, 64
N_SAMPLES = 2400  
X, y, y_mean, y_std = generate_synthetic_spatiotemporal_data(n_samples=N_SAMPLES, seq_len=SEQ_LEN, H=H, W=W)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# -----------------------
# Model builders

def build_cnn_lstm(seq_len=SEQ_LEN, H=H, W=W, channels=1):
    """
    TimeDistributed CNN -> LSTM -> Dense
    Input shape: (seq_len, H, W, channels)
    """
    from tensorflow.keras import Input, Model
    inp = Input(shape=(seq_len, H, W, channels))

    # TimeDistributed small CNN
    td = layers.TimeDistributed(layers.Conv2D(16, (3,3), activation='relu', padding='same'))(inp)
    td = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(td)
    td = layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', padding='same'))(td)
    td = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(td)
    td = layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu', padding='same'))(td)
    td = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(td)
    td = layers.TimeDistributed(layers.Flatten())(td)  # shape -> (seq_len, features)

    lstm = layers.LSTM(128, activation='tanh')(td)
    dense = layers.Dense(64, activation='relu')(lstm)
    out = layers.Dense(1, activation='linear')(dense)
    model = Model(inputs=inp, outputs=out, name='CNN-LSTM')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_conv_lstm(seq_len=SEQ_LEN, H=H, W=W, channels=1):
    """
    ConvLSTM2D model directly on the spatio-temporal input.
    Input shape: (seq_len, H, W, channels)
    """
    from tensorflow.keras import Input, Model
    inp = Input(shape=(seq_len, H, W, channels))
    x = layers.ConvLSTM2D(32, (3,3), padding='same', return_sequences=True, activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM2D(64, (3,3), padding='same', return_sequences=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)
    model = Model(inputs=inp, outputs=out, name='ConvLSTM')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_transformer(seq_len=SEQ_LEN, H=H, W=W, channels=1, embed_dim=128, num_heads=4):
    """
    CNN per frame to extract features -> Transformer Encoder across timesteps -> Dense
    Input: (seq_len, H, W, channels)
    Transformer expects (seq_len, embed_dim)
    """
    from tensorflow.keras import Input, Model
    inp = Input(shape=(seq_len, H, W, channels))

    td = layers.TimeDistributed(layers.Conv2D(16, 3, activation='relu', padding='same'))(inp)
    td = layers.TimeDistributed(layers.MaxPooling2D(2))(td)
    td = layers.TimeDistributed(layers.Conv2D(32, 3, activation='relu', padding='same'))(td)
    td = layers.TimeDistributed(layers.MaxPooling2D(2))(td)
    td = layers.TimeDistributed(layers.Conv2D(64, 3, activation='relu', padding='same'))(td)
    td = layers.TimeDistributed(layers.MaxPooling2D(2))(td)
    td = layers.TimeDistributed(layers.Flatten())(td)  # (batch, seq_len, features)

    # project to embed_dim
    proj = layers.TimeDistributed(layers.Dense(embed_dim))(td)

    x = proj
    for _ in range(2):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        ff = layers.Dense(embed_dim*2, activation='relu')(x)
        ff = layers.Dense(embed_dim)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    # pool across time
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=inp, outputs=out, name='CNN-Transformer')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

cnn_lstm_model = build_cnn_lstm()
conv_lstm_model = build_conv_lstm()
cnn_transformer_model = build_cnn_transformer()

cnn_lstm_model.summary()
conv_lstm_model.summary()
cnn_transformer_model.summary()

# -----------------------
# Training utilities
EPOCHS = 15
BATCH_SIZE = 32

def train_and_evaluate(model, X_tr, y_tr, X_te, y_te, epochs=EPOCHS, batch_size=BATCH_SIZE):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=2)

    y_pred = model.predict(X_te).squeeze()
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    return history, y_pred, {'mae': mae, 'rmse': rmse, 'r2': r2}

# -----------------------
# Train CNN-LSTM
print("\n--- Training CNN-LSTM ---")
hist1, y_pred1, metrics1 = train_and_evaluate(cnn_lstm_model, X_train, y_train, X_test, y_test)
print("CNN-LSTM metrics:", metrics1)

# Train ConvLSTM
print("\n--- Training ConvLSTM ---")
hist2, y_pred2, metrics2 = train_and_evaluate(conv_lstm_model, X_train, y_train, X_test, y_test)
print("ConvLSTM metrics:", metrics2)

# Train CNN-Transformer
print("\n--- Training CNN-Transformer ---")
hist3, y_pred3, metrics3 = train_and_evaluate(cnn_transformer_model, X_train, y_train, X_test, y_test)
print("CNN-Transformer metrics:", metrics3)

def plot_predictions(y_true, preds, title):
    plt.figure(figsize=(8,4))
    plt.plot(y_true, label='True', alpha=0.7)
    plt.plot(preds, label='Pred', alpha=0.7)
    plt.title(title)
    plt.xlabel('Test sample index')
    plt.ylabel('Normalized target')
    plt.legend()
    plt.show()

plot_predictions(y_test, y_pred1, f'CNN-LSTM (MAE={metrics1["mae"]:.3f}, RMSE={metrics1["rmse"]:.3f}, R2={metrics1["r2"]:.3f})')
plot_predictions(y_test, y_pred2, f'ConvLSTM (MAE={metrics2["mae"]:.3f}, RMSE={metrics2["rmse"]:.3f}, R2={metrics2["r2"]:.3f})')
plot_predictions(y_test, y_pred3, f'CNN-Transformer (MAE={metrics3["mae"]:.3f}, RMSE={metrics3["rmse"]:.3f}, R2={metrics3["r2"]:.3f})')

# Scatter plots
def scatter_true_vs_pred(y_true, preds, title):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, preds, alpha=0.4)
    mn, mx = min(y_true.min(), preds.min()), max(y_true.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

scatter_true_vs_pred(y_test, y_pred1, 'CNN-LSTM True vs Pred')
scatter_true_vs_pred(y_test, y_pred2, 'ConvLSTM True vs Pred')
scatter_true_vs_pred(y_test, y_pred3, 'CNN-Transformer True vs Pred')

import pandas as pd
df = pd.DataFrame({
    'model': ['CNN-LSTM', 'ConvLSTM', 'CNN-Transformer'],
    'mae': [metrics1['mae'], metrics2['mae'], metrics3['mae']],
    'rmse': [metrics1['rmse'], metrics2['rmse'], metrics3['rmse']],
    'r2': [metrics1['r2'], metrics2['r2'], metrics3['r2']],
})
print("\nSummary metrics:\n", df)


