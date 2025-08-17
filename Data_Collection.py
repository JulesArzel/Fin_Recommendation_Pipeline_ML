import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# Fonction pour récupérer les ratios financiers
def import_data(companies, ratios):
    company_names = []

    for company, symbol in companies.items():
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                print(f"Récupération des ratios pour {company} ({symbol})...")
                ticker = yf.Ticker(symbol)  # PAS de session
                info = ticker.info
                company_names.append(company)

                # Récupérer les ratios (utiliser .get pour éviter les erreurs si la clé est absente)
                ratios["forwardPE"].append(info.get("forwardPE"))
                ratios["beta"].append(info.get("beta"))
                ratios["priceToBook"].append(info.get("priceToBook"))
                ratios["priceToSales"].append(info.get("priceToSalesTrailing12Months"))
                ratios["dividendYield"].append(info.get("dividendYield"))
                ratios["trailingEps"].append(info.get("trailingEps"))
                ratios["debtToEquity"].append(info.get("debtToEquity"))
                ratios["currentRatio"].append(info.get("currentRatio"))
                ratios["quickRatio"].append(info.get("quickRatio"))
                ratios["returnOnEquity"].append(info.get("returnOnEquity"))
                ratios["returnOnAssets"].append(info.get("returnOnAssets"))
                ratios["operatingMargins"].append(info.get("operatingMargins"))
                ratios["profitMargins"].append(info.get("profitMargins"))

                break  # sortie de boucle si succès

            except Exception as e:
                print(f"⚠️ Erreur pour {company} ({symbol}): {e}")
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"⏳ Nouvelle tentative dans {wait_time} secondes...")
                time.sleep(wait_time)

    # Création DataFrame et export
    df = pd.DataFrame(ratios, index=company_names)
    df.to_csv("ratios_compagnies.csv")
    print("Les ratios financiers ont été exportés vers 'ratios_compagnies.csv'.")

# Fonction pour récupérer les données historiques
def scrapping_data(companies):
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")

    output_folder = "Companies_historical_data"
    os.makedirs(output_folder, exist_ok=True)

    for company, symbol in companies.items():
        retry_count = 0
        max_retries = 5
        success = False

        while retry_count < max_retries and not success:
            try:
                print(f"Téléchargement des données historiques pour {company} ({symbol})...")
                ticker = yf.Ticker(symbol)  #PAS de session
                data = ticker.history(start=start_date, end=end_date)

                if data.empty or "Close" not in data.columns:
                    print(f"⚠️ Aucune donnée trouvée pour {company} ({symbol})")
                    break

                df = data[["Close"]].copy()
                df["Next Day Close"] = df["Close"].shift(-1)
                df["Rendement"] = (df["Next Day Close"] - df["Close"]) / df["Close"]
                df.dropna(inplace=True)

                file_path = os.path.join(output_folder, f"{company}_historical_data.csv")
                df.to_csv(file_path)

                print(f"Données exportées pour {company} ({symbol})")
                success = True
                time.sleep(2)

            except Exception as e:
                wait_time = 2 ** retry_count
                print(f"Erreur pour {company} ({symbol}) : {e} - nouvelle tentative dans {wait_time} s")
                time.sleep(wait_time)
                retry_count += 1

    print("Tous les fichiers ont été enregistrés dans le dossier 'Companies_historical_data'.")

# Point d’entrée principal pour test (facultatif)
if __name__ == "__main__":
    companies = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA",
    }

    ratios = {
        "forwardPE": [], "beta": [], "priceToBook": [], "priceToSales": [],
        "dividendYield": [], "trailingEps": [], "debtToEquity": [],
        "currentRatio": [], "quickRatio": [], "returnOnEquity": [],
        "returnOnAssets": [], "operatingMargins": [], "profitMargins": [],
    }

    import_data(companies, ratios)
    scrapping_data(companies)
