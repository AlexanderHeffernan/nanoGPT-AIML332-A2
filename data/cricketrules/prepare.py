import os
import tiktoken
import numpy as np

input_file_path = "cricket_rules.txt"
with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
np.array(train_ids, dtype=np.uint16).tofile("train.bin")
np.array(val_ids, dtype=np.uint16).tofile("val.bin")

# Save meta.pkl for encoding/decoding
meta = {
    "vocab_size": enc.n_vocab,
    "encode": enc.encode_ordinary,
    "decode": enc.decode
}
import pickle
with open("meta.pkl", "wb") as f:
    pickle.dump(meta, f)