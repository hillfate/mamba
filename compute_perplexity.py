import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Perplexity Computation for Mamba2")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba2-2.7b")
parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to evaluate perplexity on")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
args = parser.parse_args()

# Device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16  # Use fp16 for efficiency

# Check if the model is Mamba
is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")

# Load Tokenizer
if is_mamba:
    print("Using GPT-NeoX tokenizer for Mamba model")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
    
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

# Print model size
print(f"Model loaded: {args.model_name}")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Load dataset
print("Loading dataset...")
# dataset = load_dataset("wikitext", "wikitext-2-v1", split="validation")
dataset = load_dataset("allenai/c4", "realnewslike", split="validation[:10%]")

# Function to compute perplexity
def compute_perplexity(text):
    # Skip empty texts
    if len(text.strip()) == 0:
        print("Skipping empty text input")
        return None

    # Tokenize with padding
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Ensure input_ids is int64
    inputs["input_ids"] = inputs["input_ids"].to(torch.long)

    # Ensure non-zero sequence length
    if inputs["input_ids"].shape[1] == 0:
        print("Skipping text with empty tokenization")
        return None

    # Remove attention_mask since Mamba2 does not use it
    if "attention_mask" in inputs:
        del inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[:, :-1, :]  # Shift left for teacher forcing
    labels = inputs["input_ids"][:, 1:]  # Shift right to align predictions

    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    # Compute mean negative log-likelihood
    mean_nll = nll.mean().item()

    # Compute perplexity
    perplexity = torch.exp(torch.tensor(mean_nll)).item()
    return perplexity

# Compute perplexity on first `args.num_samples` samples
print(f"Computing perplexity on {args.num_samples} samples...")
perplexities = [compute_perplexity(text) for text in dataset["text"][:args.num_samples]]

# Remove None values before averaging
valid_perplexities = [p for p in perplexities if p is not None]

if len(valid_perplexities) == 0:
    print("No valid perplexity values computed. Check model behavior.")
    exit(1)

avg_perplexity = sum(valid_perplexities) / len(valid_perplexities)
print(f"✅ Average Perplexity: {avg_perplexity}")
