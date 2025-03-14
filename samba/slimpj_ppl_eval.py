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
    total_real_tokens = 0  # Tracks only non-padding tokens

    with torch.no_grad():
        for idx, file_path in enumerate(files):
            # Print every 100th file
            if idx % 100 == 0:
                logging.info(f"Processing file {idx+1}/{len(files)}: {file_path} at context length {max_length}")

            # MIN_LENGTH_FACTOR = 0.75  # Keep at least 75% of context length

            for json_obj in read_jsonl_zst(file_path):
                text = json_obj["text"]

                # Tokenize without padding to measure actual length
                encoded_raw = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
                # real_length = encoded_raw["input_ids"].shape[1]

                # # Compute dynamic min length threshold (e.g., 75% of context length)
                # min_required_length = int(max_length * MIN_LENGTH_FACTOR)

                # # Skip texts shorter than 75% of context length
                # if real_length < min_required_length:
                #     logging.info(f"Skipping text with {real_length} tokens (too short for context {max_length})")
                #     continue

                # Tokenize
                # encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
                input_ids = encoded_raw["input_ids"].to(device)

                # Count non-padding tokens
                real_token_mask = input_ids != tokenizer.pad_token_id  # Mask: True for real tokens, False for padding
                num_real_tokens = real_token_mask.sum().item()  # Count real tokens
                total_real_tokens += num_real_tokens  # Update global counter

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
                total_tokens += shift_labels.numel()  # This includes padding tokens

    # Compute loss per token
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    # Compute loss considering only real tokens (no padding)
    avg_real_loss = total_loss / total_real_tokens
    real_ppl = math.exp(avg_real_loss)

    # Log information about padding bias
    logging.info(f"Max Length: {max_length} | Total Tokens: {total_tokens} | Non-Pad Tokens: {total_real_tokens}")
    logging.info(f"Regular Perplexity: {perplexity:.4f} | Real Token Perplexity: {real_ppl:.4f}")

    return avg_loss, perplexity, avg_real_loss, real_ppl

# --- Run Perplexity Evaluation ---
if __name__ == "__main__":
    print("Loading model...")
    model = load_model(CHECKPOINT_PATH, CONFIG_NAME, DEVICE, DTYPE)

    # for context_length in CONTEXT_LENGTHS:
    #     print(f"Computing Perplexity at Context Length {context_length}...")
    #     ppl = compute_perplexity(model, jsonl_zst_files, tokenizer, DEVICE, max_length=context_length)
    #     print(f"Perplexity on SlimPajama (Context Length {context_length}): {ppl:.4f}")

    for context_length in CONTEXT_LENGTHS:
        print(f"Computing Perplexity at Context Length {context_length}...")
        avg_loss, ppl, avg_real_loss, real_ppl = compute_perplexity(model, jsonl_zst_files, tokenizer, DEVICE, max_length=context_length)
        print(f"Perplexity on SlimPajama (Context Length {context_length}): {ppl:.4f} | Real Token PPL: {real_ppl:.4f}")