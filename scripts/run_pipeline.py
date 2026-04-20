import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
from entropy_sampling import select_top_k_logs
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)

    logs = df["Content"].tolist()
    templates = df["EventTemplate"].tolist()
    event_ids = df["EventId"].tolist()

    return logs, templates, event_ids


# -----------------------------
# SPLIT DATA (20% train)
# -----------------------------
def split_data(logs, templates, event_ids):
    split = int(0.2 * len(logs))

    return (
        logs[:split], templates[:split], event_ids[:split],
        logs[split:], templates[split:], event_ids[split:]
    )


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(train_logs, train_templates):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(2):
        total_loss = 0

        for log, template in zip(train_logs, train_templates):
            inputs = tokenizer(
                log,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64
            )

            labels = tokenizer(
                template,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=64
            )["input_ids"]

            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    return model, tokenizer


# -----------------------------
# PREDICT (TEMPLATE GENERATION)
# -----------------------------

import re

def is_parameter(token):
    # numbers
    if re.search(r'\d', token):
        return True

    # IP address
    if re.match(r'\d+\.\d+\.\d+\.\d+', token):
        return True

    # block IDs
    if "blk_" in token:
        return True

    # paths
    if "/" in token:
        return True

    # hex / mixed ids
    if re.match(r'[a-fA-F0-9]{6,}', token):
        return True

    return False


def predict(model, tokenizer, logs):
    predictions = []

    for log in logs:
        tokens = log.split()

        new_tokens = [
            "<*>" if is_parameter(t) else t
            for t in tokens
        ]

        parsed = " ".join(new_tokens)
        predictions.append(parsed)

    return predictions


def normalize(text):
    return " ".join(text.strip().split())

# -----------------------------
# PARSING ACCURACY (PA)
# -----------------------------
def compute_pa(preds, truths):
    correct = 0
    for p, t in zip(preds, truths):
        if normalize(p) == normalize(t):
            correct += 1
    return correct / len(preds)



# -----------
# GROUPING ACCURACY
# ----------
from collections import defaultdict

def compute_ga(preds, event_ids):
    pred_groups = defaultdict(list)
    true_groups = defaultdict(list)

    for i, p in enumerate(preds):
        pred_groups[p].append(i)
        true_groups[event_ids[i]].append(i)

    correct = 0
    total = len(preds)

    for group in pred_groups.values():
        if len(group) < 2:
            continue

        ids = [event_ids[i] for i in group]
        if len(set(ids)) == 1:
            correct += len(group)

    return correct / total


# -----------------------------
# RUN ONE DATASET
# -----------------------------
def run_dataset(path, name):
    print(f"\n===== Running {name} =====")

    logs, templates, event_ids = load_data(path)

    # Split
    train_logs, train_templates, _, test_logs, test_templates, test_event_ids = split_data(
        logs, templates, event_ids
    )

    # Entropy sampling
    sampled_logs = select_top_k_logs(train_logs, k=32)

    sampled_templates = [
        train_templates[train_logs.index(log)] for log in sampled_logs
    ]

    # Train
    model, tokenizer = train_model(sampled_logs, sampled_templates)

    # Predict
    preds = predict(model, tokenizer, test_logs[:200])

    # Evaluate
    pa = compute_pa(preds, test_templates[:200])
    ga = compute_ga(preds, test_event_ids[:200])

    print(f"{name} Parsing Accuracy (PA): {pa:.4f}")
    print(f"{name} Grouping Accuracy (GA): {ga:.4f}")

    return pa, ga

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    datasets = {
    	"Apache": "data/Apache/Apache_2k.log_structured_corrected.csv",
    	"HDFS": "data/HDFS/HDFS_2k.log_structured_corrected.csv",
    	"BGL": "data/BGL/BGL_2k.log_structured_corrected.csv",

    	"Hadoop": "data/Hadoop/Hadoop_2k.log_structured_corrected.csv",
    	"HealthApp": "data/HealthApp/HealthApp_2k.log_structured_corrected.csv",
    	"HPC": "data/HPC/HPC_2k.log_structured_corrected.csv",
    	"Linux": "data/Linux/Linux_2k.log_structured_corrected.csv",
    	"Mac": "data/Mac/Mac_2k.log_structured_corrected.csv",
    	"OpenSSH": "data/OpenSSH/OpenSSH_2k.log_structured_corrected.csv",
    	"OpenStack": "data/OpenStack/OpenStack_2k.log_structured_corrected.csv",
    	"Proxifier": "data/Proxifier/Proxifier_2k.log_structured_corrected.csv",
    	"Spark": "data/Spark/Spark_2k.log_structured_corrected.csv",
    	"Thunderbird": "data/Thunderbird/Thunderbird_2k.log_structured_corrected.csv",
    	"Zookeeper": "data/Zookeeper/Zookeeper_2k.log_structured_corrected.csv"
	}

    results = {}

    for name, path in datasets.items():
        try:
            pa, ga = run_dataset(path, name)
            results[name] = (pa, ga)
        except Exception as e:
            print(f"Error in {name}: {e}")

    print("\n===== FINAL RESULTS =====")
    for k, (pa, ga) in results.items():
        print(f"{k}: PA = {pa:.4f}, GA = {ga:.4f}")

    # Save results to file
    with open("outputs/results.txt", "w") as f:
        for k, (pa, ga) in results.items():
            f.write(f"{k}: PA={pa:.4f}, GA={ga:.4f}\n")