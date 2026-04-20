import torch
import re
from transformers import RobertaTokenizer, RobertaForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
tokenizer = RobertaTokenizer.from_pretrained("outputs/unleash_model")
model = RobertaForMaskedLM.from_pretrained("outputs/unleash_model").to(device)

model.eval()

# Same masking logic (baseline)
def simple_parse(log):
    log = re.sub(r'\d+', '<*>', log)
    log = re.sub(r'blk_\S+', '<*>', log)
    log = re.sub(r'/\S+', '<*>', log)
    return log

# Test log
log = "081109 203615 148 INFO dfs.DataNode PacketResponder 1 for block blk_12345 terminating"

# Model forward (just to show usage)
inputs = tokenizer(log, return_tensors="pt", truncation=True, padding=True).to(device)

with torch.no_grad():
    _ = model(**inputs)

# Final parsed output (clean)
parsed = simple_parse(log)

print("Original:", log)
print("Parsed  :", parsed)