from transformers import AutoModelForSequenceClassification, AutoTokenizer

local_path = "./yiyanghkust_finbert-tone_results"
model_name = "yiyanghkust/finbert-tone"

# Télécharger et sauvegarder le modèle en local
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)

print("✅ Modèle sauvegardé localement !")
