import pandas as pd
import numpy as np
import glob
import ta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")


def load_and_prepare_data(file_path, n_days=30):
    df = pd.read_csv(file_path)
    df = df[['Close']].dropna()

    # Normalisation
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # Création des features X et des cibles Y
    x, y = create_target_features(scaled_data, n_days)

    # Séparation train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=0.2)

    return x_train, x_test, y_train, y_test, scaler, df['Close'].values


def create_target_features(data, n):
    x, y = [], []
    for i in range(n, len(data)):
        x.append(data[i - n:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


def evaluate_model(name, model, param_grid, x_train, x_test, y_train, y_test, scaler, close_prices, offset=30):
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(x_test)

    # Inversion de la mise à l’échelle
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)

    print(f"\n=== {name} ===")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(close_prices)), close_prices, color='red', label='Valeurs réelles')
    plt.plot(range(len(close_prices) - len(y_pred_inv), len(close_prices)), y_pred_inv, color='blue', label=f'{name} Predictions')
    plt.title(f"{name} - Valeurs réelles vs prédictions")
    plt.legend()
    plt.grid(True)

    return {'Modèle': name, 'MSE': mse, 'RMSE': rmse}


def run_all_models(file_path):
    x_train, x_test, y_train, y_test, scaler, close_prices = load_and_prepare_data(file_path)

    results = []

    results.append(evaluate_model("XGBoost", XGBRegressor(objective='reg:squarederror'),
                                  {"max_depth": [3, 5], "n_estimators": [100, 200]},
                                  x_train, x_test, y_train, y_test, scaler, close_prices))

    results.append(evaluate_model("Random Forest", RandomForestRegressor(),
                                  {"n_estimators": [100, 200]},
                                  x_train, x_test, y_train, y_test, scaler, close_prices))

    results.append(evaluate_model("KNN", KNeighborsRegressor(),
                                  {"n_neighbors": [3, 5, 7]},
                                  x_train, x_test, y_train, y_test, scaler, close_prices))

    results.append(evaluate_model("Régression Linéaire", LinearRegression(),
                                  {},
                                  x_train, x_test, y_train, y_test, scaler, close_prices))

    results_df = pd.DataFrame(results)
    print("\n=== Résumé des performances ===")
    print(results_df)


if __name__ == "__main__":
    run_all_models("Companies_historical_data/Apple_historical_data.csv")