import torch
import torch.nn as nn
import math
import os
import json
import random
import zstandard as zstd
from pathlib import Path
from transformers import AutoTokenizer
from lit_gpt.model import GPT, Config

import logging
from datetime import datetime

# --- Configure Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",  # Custom datetime format
)

# --- CONFIGURATION ---
CHECKPOINT_PATH = "/network/rit/lab/yinlab/ydeng/results/mambaResults/sam_mamba/out/tsz512x4k_20B_bigram_Mamba_430M/iter-144000-ckpt.pth"  
CONFIG_NAME = "bigram_Mamba_430M"
SLIMPAJAMA_PATH = "/network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/SlimPajama-627B/validation"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16  # Adjust based on your hardware
CONTEXT_LENGTHS = [4096, 8192, 16384]  # Context lengths to test

# --- Load Model ---
def load_model(checkpoint_path, config_name, device, dtype):
    config = Config.from_name(config_name)
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model

# --- Load Tokenizer ---
tokenizer_name = "meta-llama/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

# # --- Get List of All `.jsonl.zst` Files ---
# def get_jsonl_zst_files(dataset_path):
#     files = []
#     for chunk in sorted(os.listdir(dataset_path)):  # Iterate through validation chunks
#         chunk_path = os.path.join(dataset_path, chunk)
#         if os.path.isdir(chunk_path):  # Ensure it's a folder
#             files.extend(sorted(Path(chunk_path).glob("*.jsonl.zst")))  # Find all `.jsonl.zst` files
#     return files

# --- Get List of 1000 Randomly Sampled `.jsonl.zst` Files ---
def get_jsonl_zst_files(dataset_path, sample_size=1000, seed=42):
    """Fetch all `.jsonl.zst` files and randomly sample `sample_size` of them."""
    files = []
    for chunk in sorted(os.listdir(dataset_path)):  # Iterate through validation chunks
        chunk_path = os.path.join(dataset_path, chunk)
        if os.path.isdir(chunk_path):  # Ensure it's a folder
            files.extend(sorted(Path(chunk_path).glob("*.jsonl.zst")))  # Find all `.jsonl.zst` files

    # Ensure we do not sample more than available files
    sample_size = min(sample_size, len(files))

    # Randomly sample `sample_size` files
    random.seed(seed)  # Ensure reproducibility
    sampled_files = random.sample(files, k=sample_size)

    return sampled_files

jsonl_zst_files = get_jsonl_zst_files(SLIMPAJAMA_PATH)

def read_jsonl_zst(file_path):
    """Properly decompress `.zst` and read JSON lines using a stream."""
    with open(file_path, "rb") as f:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(f) as reader:
            text_stream = reader.read().decode("utf-8")  # Read all text
            for line in text_stream.split("\n"):  # Split lines manually
                line = line.strip()
                if line:  # Ignore empty lines
                    try:
                        json_obj = json.loads(line)  # Parse JSON
                        yield json_obj
                    except json.JSONDecodeError as e:
                        print(f"JSON Decode Error in {file_path}: {e}")
                        continue  # Skip bad lines

# --- Compute Perplexity ---
def compute_perplexity(model, files, tokenizer, device, max_length):
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for idx, file_path in enumerate(files):
            # Print every 100th file
            if idx % 10 == 0:
                logging.info(f"Processing file {idx+1}/{len(files)}: {file_path} at context length {max_length}")

            for json_obj in read_jsonl_zst(file_path):
                text = json_obj["text"]  # Extract raw text

                # Tokenize
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
                input_ids = encoded["input_ids"].to(device)

                # Ensure labels are correctly aligned
                input_tokens = input_ids[:, :-1].contiguous()  # Remove last token from inputs
                labels = input_ids[:, 1:].contiguous()  # Shift labels to match logits

                # Forward pass
                logits = model(input_tokens).logits  # Shape: (batch_size, seq_len-1, vocab_size)
                shift_logits = logits.contiguous()  # Ensure contiguous memory layout
                shift_labels = labels.contiguous()  # Ensure contiguous memory layout

                # Compute loss
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                total_loss += loss.item()
                total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# --- Run Perplexity Evaluation ---
if __name__ == "__main__":
    print("Loading model...")
    model = load_model(CHECKPOINT_PATH, CONFIG_NAME, DEVICE, DTYPE)

    for context_length in CONTEXT_LENGTHS:
        print(f"Computing Perplexity at Context Length {context_length}...")
        ppl = compute_perplexity(model, jsonl_zst_files, tokenizer, DEVICE, max_length=context_length)
        print(f"Perplexity on SlimPajama (Context Length {context_length}): {ppl:.4f}")