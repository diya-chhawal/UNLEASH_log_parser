import math
from collections import Counter

def compute_entropy(log):
    tokens = log.split()
    counts = Counter(tokens)
    total = len(tokens)

    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p + 1e-9)

    return entropy


def select_top_k_logs(logs, k=32):
    scored_logs = []

    for log in logs:
        ent = compute_entropy(log)
        scored_logs.append((log, ent))

    # Sort by entropy (high → low)
    scored_logs.sort(key=lambda x: x[1], reverse=True)

    # Return top-k logs
    return [log for log, _ in scored_logs[:k]]


if __name__ == "__main__":
    from load_logs import load_logs

    logs = load_logs("data/HDFS/HDFS.log", num_lines=1000)

    selected = select_top_k_logs(logs, k=32)

    print("Top entropy logs:")
    for log in selected[:5]:
        print(log)