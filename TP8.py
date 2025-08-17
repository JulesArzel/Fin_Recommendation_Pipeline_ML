import json
import os
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yfinance as yf
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from collections import defaultdict
from bisect import bisect_left


def get_texts_timestamps(news_path):
    with open(news_path, "r") as f:
        news_data = json.load(f)

    eastern = pytz.timezone("America/New_York")
    texts = []
    timestamps = []

    for date, articles in news_data.items():
        for article in articles:
            ts_utc = datetime.fromisoformat(article["publishedAt"].replace("Z", ""))
            ts_local = ts_utc.astimezone(eastern)
            ts_rounded = ts_local.replace(minute=0, second=0, microsecond=0)
            full_text = f"{article.get('title', '')} {article.get('description', '')}"
            texts.append(full_text.strip())
            timestamps.append(ts_rounded)

    return texts, timestamps


def get_sentiments(model_path, texts):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    sentiments = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        sentiments.append(pred)

    return sentiments


def align_timestamps(timestamps):
    aligned = []
    for ts in timestamps:
        if 9 <= ts.hour < 15:
            aligned.append(ts.replace(minute=0, second=0, microsecond=0))
        elif 15 <= ts.hour < 24:
            aligned.append(ts.replace(hour=15, minute=0, second=0, microsecond=0))
        else:  # 00h à 9h
            aligned.append((ts - timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0))
    return aligned


def find_nearest_timestamp(ts, available_ts):
    pos = bisect_left(available_ts, ts)
    if pos == 0:
        return available_ts[0]
    if pos == len(available_ts):
        return available_ts[-1]
    before = available_ts[pos - 1]
    after = available_ts[pos]
    return before if abs(ts - before) < abs(ts - after) else after


def plot_comparison(df, sentiments_a, sentiments_b, timestamps, title_a, title_b):
    aligned_ts = align_timestamps(timestamps)
    grouped_a = defaultdict(list)
    grouped_b = defaultdict(list)

    for ts, sa, sb in zip(aligned_ts, sentiments_a, sentiments_b):
        grouped_a[ts].append(sa)
        grouped_b[ts].append(sb)

    available_ts = list(df['Datetime'])

    def plot_sub(ax, grouped_sent, title):
        ax.plot(df['Datetime'], df['Close'], color='black', label='Price')
        for ts, sentiments in grouped_sent.items():
            nearest_ts = find_nearest_timestamp(ts, available_ts)
            matching_row = df[df['Datetime'] == nearest_ts]
            if not matching_row.empty:
                base_price = matching_row['Close'].values[0]
                for i, s in enumerate(sentiments):
                    if s == 2:
                        color = 'green'
                    elif s == 1:
                        color = 'orange'
                    else:
                        color = 'red'
                    ax.scatter(nearest_ts, base_price + i * 0.2, color=color, s=40)
        ax.set_title(title)
        ax.grid(True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    plot_sub(ax1, grouped_a, title_a)
    plot_sub(ax2, grouped_b, title_b)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Neutral', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='black', lw=2, label='Stock Price')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.show()


# Fonction principale à lancer
def run_analysis(company_ticker, model_a_path, model_b_path):
    print(f"📊 Chargement des données boursières pour {company_ticker}...")
    ticker = yf.Ticker(company_ticker)
    df = ticker.history(start="2025-01-01", interval="60m").reset_index()

    json_path = os.path.join("news_data", f"{company_ticker}_news.json")
    print(f" Chargement des actualités depuis {json_path}")
    texts, timestamps = get_texts_timestamps(json_path)

    print(" Analyse des sentiments avec le modèle A...")
    sentiments_a = get_sentiments(model_a_path, texts)

    print(" Analyse des sentiments avec le modèle B...")
    sentiments_b = get_sentiments(model_b_path, texts)

    print(" Génération du graphique...")
    plot_comparison(df, sentiments_a, sentiments_b, timestamps, "Modèle A", "Modèle B")


# Lancement
if __name__ == "__main__":
    run_analysis("AAPL", "ProsusAI/finbert", "./yiyanghkust_finbert-tone_results")
