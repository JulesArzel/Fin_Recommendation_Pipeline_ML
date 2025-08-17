import pandas as pd
import glob
import numpy as np
import ta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import shap
import warnings
warnings.filterwarnings("ignore")


def create_labels(file_path):
    df = pd.read_csv(file_path)
    df['Close_Horizon'] = df['Close'].shift(-20)
    df['Horizon_Return'] = (df['Close_Horizon'] - df['Close']) / df['Close']
    df['Label'] = df['Horizon_Return'].apply(lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1))
    return df


def compute_features(df):
    df['SMA 20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA 20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI 14'] = ta.momentum.rsi(df['Close'], window=14)

    macd = ta.trend.macd(df['Close'])
    df['MACD'] = macd
    df['MACD Signal'] = ta.trend.macd_signal(df['Close'])

    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20)
    df['Bollinger High'] = bollinger.bollinger_hband()
    df['Bollinger Low'] = bollinger.bollinger_lband()

    df['Rolling Volatility 20'] = df['Close'].rolling(window=20).std()
    df['ROC 10'] = ta.momentum.roc(df['Close'], window=10)

    df.dropna(inplace=True)
    return df



def prepare_dataset(folder="Companies_historical_data"):
    all_data = []
    for file in glob.glob(f"{folder}/*.csv"):
        df = create_labels(file)
        df = compute_features(df)
        all_data.append(df)
    full_df = pd.concat(all_data)
    full_df.dropna(inplace=True)
    return full_df


def process_for_classification(df):
    drop_cols = ['Label', 'Close_Horizon', 'Next Day Close', 'Rendement', 'Horizon_Return']

    features = [col for col in df.columns if col not in drop_cols and df[col].dtype != 'object']
    X = df[features]
    y = df['Label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, shuffle=True, random_state=42), X, features


def run_classifier(name, model, param_grid, X_train, X_test, y_train, y_test, X, features):
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"\n=== Résultats {name} ===")
    print(classification_report(y_test, y_pred))
    if hasattr(best_model, 'predict_proba'):
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values[:, :, 2], X_test, feature_names=features, show=False)
        shap.summary_plot(shap_values[:, :, 0], X_test, feature_names=features, show=False)
    return accuracy_score(y_test, y_pred)


def compare_models():
    df = prepare_dataset()
    (X_train, X_test, y_train, y_test), X, features = process_for_classification(df)

    scores = {}
    scores['XGBoost'] = run_classifier("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                                       {"max_depth": [3, 5], "n_estimators": [100, 200]},
                                       X_train, X_test, y_train, y_test, X, features)

    scores['Random Forest'] = run_classifier("Random Forest", RandomForestClassifier(),
                                            {"n_estimators": [100, 200]},
                                            X_train, X_test, y_train, y_test, X, features)

    scores['KNN'] = run_classifier("KNN", KNeighborsClassifier(),
                                   {"n_neighbors": [3, 5, 7]},
                                   X_train, X_test, y_train, y_test, X, features)

    scores['SVM'] = run_classifier("SVM", SVC(probability=True),
                                   {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
                                   X_train, X_test, y_train, y_test, X, features)

    scores['Logistic Regression'] = run_classifier("Logistic Regression", LogisticRegression(max_iter=1000),
                                                  {"C": [0.1, 1, 10]},
                                                  X_train, X_test, y_train, y_test, X, features)

    print("\n=== Tableau des performances ===")
    for model, score in scores.items():
        print(f"{model}: {score:.4f}")


if __name__ == "__main__":
    compare_models()
