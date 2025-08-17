from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np



# 1. Téléchargement et concaténation des datasets financiers

def load_financial_datasets(test_size=0.2, seed=42):
    ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")  # a un split train + test
    ds2 = load_dataset("nickmuchi/financial-classification")         # uniquement train

    ds1 = ds1.rename_column("text", "sentence")
    ds2 = ds2.rename_column("labels", "label")

    combined_train = concatenate_datasets([ds1["train"], ds2["train"]])
    
    # Création manuelle d’un test set
    train_test_split_result = combined_train.train_test_split(test_size=test_size, seed=seed)

    return DatasetDict({
        "train": train_test_split_result["train"],
        "test": train_test_split_result["test"]
    })



# 2. Tokenisation du dataset

def tokenize_dataset(dataset, tokenizer):
    def preprocess(examples):
        texts = examples["sentence"]
        texts = [text if isinstance(text, str) else "" for text in texts]
        return tokenizer(texts, truncation=True, padding=True)

    tokenized_train = dataset["train"].map(preprocess, batched=True)
    tokenized_test = dataset["test"].map(preprocess, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_train, tokenized_test



# 3. Métriques d'évaluation personnalisées

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# 4. Entraînement du modèle

def train_model(model_name, dataset, batch_size=8, num_epochs=2):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    tokenized_train, tokenized_test = tokenize_dataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        report_to="none",
        disable_tqdm=False,
        logging_first_step=True
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )


    trainer.train()
    evaluation = trainer.evaluate()
    print(f"\n Évaluation du modèle {model_name} :")
    print(evaluation)
    return model


if __name__ == "__main__":
    dataset = load_financial_datasets()

    print("\n--- Finetuning BERT de base ---")
    bert_model = train_model("bert-base-uncased", dataset)

    print("\n--- Finetuning FinBERT ---")
    finbert_model = train_model("yiyanghkust/finbert-tone", dataset)
