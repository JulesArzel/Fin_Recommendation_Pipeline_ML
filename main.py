import os
import json
from datetime import datetime, timedelta
from statistics import mode
import model_import

# TPs importés
import TP1
import TP2
import TP3
import TP4
import TP5
import TP6
import TP7
import TP8

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor


def aggregate_recommendations(companies,
                              similar,
                              cls_signal,
                              reg_signal,
                              sentiment_signal,
                              deep_signal):
    """
    Stratégie d’agrégation :
    - BUY si Classif = 2 ET Sentiment = 2 ET LSTM RMSE < XGBoost RMSE
    - SELL si Classif = 0 ET Sentiment = 0 ET LSTM RMSE < XGBoost RMSE
    - HOLD sinon
    """
    rec = {}
    for c in companies:
        classif = cls_signal[c]
        sentiment = sentiment_signal[c]
        rmse_classic = reg_signal[c]
        rmse_lstm = deep_signal.get(c, 9999)

        if classif == 2 and sentiment == 2 and rmse_lstm < rmse_classic:
            rec[c] = "BUY"
        elif classif == 0 and sentiment == 0 and rmse_lstm < rmse_classic:
            rec[c] = "SELL"
        else:
            rec[c] = "HOLD"
    return rec


def main():
    # ---------- 1. Données brutes (TP1) ----------
    ratios = {
        "forwardPE": [], "beta": [], "priceToBook": [], "priceToSales": [],
        "dividendYield": [], "trailingEps": [], "debtToEquity": [],
        "currentRatio": [], "quickRatio": [], "returnOnEquity": [],
        "returnOnAssets": [], "operatingMargins": [], "profitMargins": [],
    }
    companies = {
        "Apple": "AAPL", "Microsoft": "MSFT", "Amazon": "AMZN",
        "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA",
    }

    TP1.import_data(companies, ratios)
    TP1.scrapping_data(companies)

    # ---------- 2. Clustering (TP2) ----------
    df_ratios = TP2.read_data("ratios_compagnies.csv")
    df_perf, pc = TP2.preprocess_for_financial_clustering(df_ratios)
    df_perf = TP2.do_kmeans_clustering(df_perf, pc)
    similar = {}
    for c in df_perf["company"].values:
        cl = df_perf.loc[df_perf["company"] == c, "Cluster"].iloc[0]
        group = df_perf.loc[df_perf["Cluster"] == cl, "company"].tolist()
        group.remove(c)
        similar[c] = group

    # ---------- 3. Classification BUY/HOLD/SELL (TP3) ----------
    cls_signal = {}
    for comp in companies:
        path = os.path.join("Companies_historical_data", f"{comp}_historical_data.csv")
        df = TP3.create_labels(path)
        df = TP3.compute_features(df)
        drop = ['Label','Close_Horizon','Next Day Close','Rendement','Horizon_Return']
        feats = [col for col in df.columns if col not in drop and df[col].dtype != 'object']
        X, y = df[feats], df["Label"]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        X_tr, X_last = Xs[:-1], Xs[-1].reshape(1, -1)
        y_tr = y.iloc[:-1]
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_tr, y_tr)
        cls_signal[comp] = int(clf.predict(X_last)[0])

    # ---------- 4. Régression classique (TP4) ----------
    reg_signal = {}
    for comp in companies:
        path = os.path.join("Companies_historical_data", f"{comp}_historical_data.csv")
        x_tr, x_te, y_tr, y_te, scaler, close = TP4.load_and_prepare_data(path, n_days=30)
        res = TP4.evaluate_model(
            "XGBoost",
            XGBRegressor(objective='reg:squarederror'),
            {"max_depth": [3, 5], "n_estimators": [100, 200]},
            x_tr, x_te, y_tr, y_te,
            scaler, close
        )
        reg_signal[comp] = res["RMSE"]

    # ---------- 5. Modèles profonds (TP5) ----------
    deep_learning_rmse = {}
    for comp in companies:
        path = os.path.join("Companies_historical_data", f"{comp}_historical_data.csv")
        results = TP5.run_all_models(path)  # Retourne dict {"MLP": ..., "RNN": ..., "LSTM": ...}
        deep_learning_rmse[comp] = results.get("LSTM", 9999)

    # ---------- 6. Récupération des news (TP6) ----------
    for name, ticker in companies.items():
        TP6.get_news_by_date(name, ticker, days=7)

    # ---------- 7. Fine-tuning des classifieurs (TP7) ----------
    dataset = TP7.load_financial_datasets()
    #TP7.train_model("bert-base-uncased", dataset, batch_size=16, num_epochs=2)
    #TP7.train_model("yiyanghkust/finbert-tone", dataset, batch_size=16, num_epochs=2)

    # ---------- 8. Analyse de sentiment (TP8) ----------
    sentiment_signal = {}
    news_headlines = {}
    for name, ticker in companies.items():
        json_path = os.path.join("news_data", f"{ticker}_news.json")
        texts, _ = TP8.get_texts_timestamps(json_path)
        sents = TP8.get_sentiments("yiyanghkust/finbert-tone", texts)
        sentiment_signal[name] = int(mode(sents)) if sents else 1
        news_headlines[name] = texts

    # ---------- 9. Agrégation des signaux ----------
    recommendations = aggregate_recommendations(companies, similar,
                                                cls_signal,
                                                reg_signal,
                                                sentiment_signal,
                                                deep_learning_rmse)

    out = []
    for c in companies:
        out.append({
            "Company": c,
            "Advice": recommendations[c],
            "Similar": similar.get(c, []),
            "Classif": cls_signal[c],
            "Forecast_RMSE": reg_signal[c],
            "RMSE_LSTM": deep_learning_rmse.get(c),
            "Sentiment": sentiment_signal[c],
            "News": news_headlines[c]
        })

    with open("recommendations.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("✅ Pipeline terminée ! Résultats dans recommendations.json")


if __name__ == "__main__":
    main()
