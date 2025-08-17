import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, Sequential, optimizers


# 1. Création du dataset pour la régression
def create_dataset(file_path, n_days=30):
    df = pd.read_csv(file_path)
    df = df[['Close']].dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(n_days, len(scaled_data)):
        X.append(scaled_data[i - n_days:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # pour RNN/LSTM

    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[:split], y[split:], scaler, df['Close'].values


# 2.1 Création des modèles
def build_mlp_model(input_shape, hidden_units=[64, 32], activation='relu', learning_rate=0.001):
    model = Sequential()
    model.add(layers.Input(shape=input_shape))
    for units in hidden_units:
        model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.Adam(learning_rate), loss='mean_squared_error')
    return model


def build_rnn_model(input_shape, hidden_units=32, activation='tanh', learning_rate=0.001):
    model = Sequential()
    model.add(layers.SimpleRNN(hidden_units, activation=activation, input_shape=input_shape))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.Adam(learning_rate), loss='mean_squared_error')
    return model


def build_lstm_model(input_shape, hidden_units=32, activation='tanh', learning_rate=0.001):
    model = Sequential()
    model.add(layers.LSTM(hidden_units, activation=activation, input_shape=input_shape))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.Adam(learning_rate), loss='mean_squared_error')
    return model


# 2.2 Entrainement des modèles
def train_model(model_type, X_train, y_train, input_shape, epochs=10, batch_size=32, hidden_units=[64, 32], learning_rate=0.001):
    if model_type == "MLP":
        X_train_flat = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        model = build_mlp_model(X_train_flat.shape[1:], hidden_units, learning_rate=learning_rate)
        model.fit(X_train_flat, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    elif model_type == "RNN":
        model = build_rnn_model(input_shape, hidden_units[0], learning_rate=learning_rate)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape, hidden_units[0], learning_rate=learning_rate)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    else:
        raise ValueError("Modèle non reconnu")
    return model


# 2.3 Prédiction et évaluation
def predict_model(model, X_test, y_test, scaler, model_type="", is_mlp=False):
    if is_mlp:
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

    print(f"\n=== Résultats {model_type} ===")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("\nPrédictions vs Réel :")
    for i in range(min(10, len(y_pred_inv))):
        print(f"Prédit: {y_pred_inv[i][0]:.2f}, Réel: {y_test_inv[i][0]:.2f}")

    return y_pred_inv, y_test_inv, mae, rmse


# 2.4 Visualisation
def plot_predictions(y_test_inv, y_pred_inv, model_name=""):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='Valeurs réelles')
    plt.plot(y_pred_inv, label='Valeurs prédites')
    plt.title(f'Comparaison Prédictions vs Réel ({model_name})')
    plt.xlabel('Temps')
    plt.ylabel('Prix')
    plt.legend()
    plt.grid(True)


# 3. Comparaison des modèles
def run_all_models(file_path):
    X_train, X_test, y_train, y_test, scaler, close = create_dataset(file_path)
    input_shape = X_train.shape[1:]
    results = []

    for model_type in ["MLP", "RNN", "LSTM"]:
        is_mlp = (model_type == "MLP")
        model = train_model(model_type, X_train, y_train, input_shape, epochs=20, hidden_units=[64, 32])
        y_pred_inv, y_test_inv, mae, rmse = predict_model(model, X_test, y_test, scaler, model_type, is_mlp)
        plot_predictions(y_test_inv, y_pred_inv, model_type)
        results.append({"Modèle": model_type, "MAE": mae, "RMSE": rmse})

    results_df = pd.DataFrame(results)
    print("\n=== Tableau des performances ===")
    print(results_df)
    return {r["Modèle"]: r["RMSE"] for r in results}



if __name__ == "__main__":
    run_all_models("Companies_historical_data/Apple_historical_data.csv")