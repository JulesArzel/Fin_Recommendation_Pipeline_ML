import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from tabulate import tabulate
import warnings

warnings.simplefilter(action="ignore", category=Warning)


def print_table(df):
    print(tabulate(df, headers="keys", tablefmt="psql"))


def read_data(path):
    df = pd.read_csv(path)
    df.rename(columns={"Unnamed: 0": "company"}, inplace=True)
    return df


def preprocess_for_financial_clustering(df):
    perf_columns = [
        "company",
        "forwardPE",
        "beta",
        "priceToBook",
        "returnOnEquity",
        "returnOnAssets",
        "operatingMargins",
        "profitMargins",
    ]
    cols_to_scale = perf_columns[1:]
    df_perf = df[perf_columns].dropna()
    scaler = StandardScaler()
    df_perf[cols_to_scale] = scaler.fit_transform(df_perf[cols_to_scale])
    return df_perf, cols_to_scale


def preprocess_risk_data(df):
    risk_columns = [
        "company",
        "debtToEquity",
        "currentRatio",
        "quickRatio",
        "dividendYield"
    ]
    cols_to_scale = risk_columns[1:]
    df_risk = df[risk_columns].dropna()
    scaler = StandardScaler()
    df_risk[cols_to_scale] = scaler.fit_transform(df_risk[cols_to_scale])
    return df_risk, cols_to_scale


def elbow_method(df, col):
    inertias = []
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[col])
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 21), inertias, marker='o')
    plt.title("Méthode du coude")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Inertie")
    plt.grid(True)
    plt.show()


def do_kmeans_clustering(df, col):
    model = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = model.fit_predict(df[col])
    print("\nCaractéristiques moyennes de chaque cluster :")
    print_table(df.groupby("Cluster")[col].mean())
    return df


def tsne_visualization(df, col):
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    df['TSNE1'], df['TSNE2'] = tsne.fit_transform(df[col]).T

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="Cluster", palette="Set1", s=100)
    for i, row in df.iterrows():
        plt.text(row.TSNE1, row.TSNE2, row.company, fontsize=8)
    plt.title("t-SNE - Clustering KMeans")
    plt.grid(True)
    plt.show()


def do_hierarchical_clustering(df, col):
    model = AgglomerativeClustering(n_clusters=4)
    df['Cluster'] = model.fit_predict(df[col])
    print("\nClusters hiérarchiques :")
    print_table(df.groupby("Cluster")[col].mean())
    return df


def plot_dendrogram(df, col):
    linked = linkage(df[col], method='ward')
    plt.figure(figsize=(12, 6))
    dendrogram(linked, labels=df['company'].values, leaf_rotation=90)
    plt.title("Dendrogramme du clustering hiérarchique")
    plt.xlabel("Entreprises")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()


def daily_returns_clustering(folder_path="Companies_historical_data"):
    rendement_dict = {}
    for file in glob.glob(f"{folder_path}/*.csv"):
        df = pd.read_csv(file)
        company = file.split("/")[-1].replace("_historical_data.csv", "")
        rendement_dict[company] = df["Rendement"]

    rendement_df = pd.DataFrame(rendement_dict)
    rendement_df.fillna(rendement_df.mean(), inplace=True)

    corr = rendement_df.corr()
    linked = linkage(corr, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(linked, labels=corr.columns, leaf_rotation=90)
    plt.title("Dendrogramme basé sur les corrélations de rendement")
    plt.grid(True)
    plt.show()


def compare_algorithms(df, col):
    models = {
        "KMeans": KMeans(n_clusters=4),
        "Agglomerative": AgglomerativeClustering(n_clusters=4),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=3),
        "Spectral": SpectralClustering(n_clusters=4, assign_labels='discretize', random_state=42)
    }
    results = []
    for name, model in models.items():
        try:
            labels = model.fit_predict(df[col])
            score = silhouette_score(df[col], labels)
            results.append({"Algorithme": name, "Silhouette Score": score})
        except Exception as e:
            results.append({"Algorithme": name, "Silhouette Score": f"Erreur: {e}"})

    print("\nComparaison des algorithmes :")
    print_table(pd.DataFrame(results))


def main():
    df = read_data("ratios_compagnies.csv")

    df_perf, perf_cols = preprocess_for_financial_clustering(df)
    elbow_method(df_perf, perf_cols)
    df_perf = do_kmeans_clustering(df_perf, perf_cols)
    tsne_visualization(df_perf, perf_cols)
    compare_algorithms(df_perf, perf_cols)

    df_risk, risk_cols = preprocess_risk_data(df)
    df_risk = do_hierarchical_clustering(df_risk, risk_cols)
    plot_dendrogram(df_risk, risk_cols)
    compare_algorithms(df_risk, risk_cols)

    daily_returns_clustering()


if __name__ == "__main__":
    main()
