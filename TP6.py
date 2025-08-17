import requests
import json
import os
from datetime import datetime, timedelta

# 🔐 À remplacer par votre clé personnelle
API_KEY = "da0bfced722f447e9e508973412da9cb"

NEWS_SOURCES = "financial-post, the-wall-street-journal, bloomberg, the-washington-post, australian-financial-review, bbc-news, cnn"
NEWS_FOLDER = "news_data"
os.makedirs(NEWS_FOLDER, exist_ok=True)

# 🔁 Correspondance nom complet → ticker boursier
company_map = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA"
}

def get_date_range(days=10):
    last_day = datetime.today().strftime('%Y-%m-%d')
    first_day = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    return first_day, last_day


def load_existing_news(ticker):
    file_path = os.path.join(NEWS_FOLDER, f"{ticker}_news.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        return {}


def save_news(ticker, news_dict):
    file_path = os.path.join(NEWS_FOLDER, f"{ticker}_news.json")
    with open(file_path, "w") as f:
        json.dump(news_dict, f, indent=4)


def get_news_by_date(company_name, ticker, days=10):
    first_day, last_day = get_date_range(days)
    url = 'https://newsapi.org/v2/everything'

    params = {
        "q": company_name,
        "sources": NEWS_SOURCES,
        "apiKey": API_KEY,
        "language": "en",
        "pageSize": 100,
        "from": first_day,
        "to": last_day,
    }

    response = requests.get(url, params=params)
    news_dict = load_existing_news(ticker)

    if response.status_code == 200:
        articles = response.json().get("articles", [])
        added = 0
        for article in articles:
            title = article.get("title", "") or ""
            description = article.get("description", "") or ""
            published_at = article.get("publishedAt", "")
            source_name = article.get("source", {}).get("name", "")

            # Vérifie que l'article parle bien de l'entreprise
            if company_name.lower() in (title + description).lower():
                date = published_at.split("T")[0]
                if date not in news_dict:
                    news_dict[date] = []

                # Évite les doublons par titre
                if not any(existing["title"] == title for existing in news_dict[date]):
                    news_dict[date].append({
                        "title": title,
                        "description": description,
                        "source": source_name,
                        "publishedAt": published_at
                    })
                    added += 1

        save_news(ticker, news_dict)
        print(f"✅ {added} articles ajoutés pour {company_name} ({ticker}).")
    else:
        print(f"❌ Erreur API pour {company_name} : {response.status_code}")


# 🔁 Boucle principale
if __name__ == "__main__":
    for name, ticker in company_map.items():
        get_news_by_date(name, ticker)
