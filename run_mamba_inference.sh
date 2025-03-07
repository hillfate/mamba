#!/bin/bash

# Default values
MODEL_NAME="state-spaces/mamba2-2.7b"
TOPP=0.9
TEMPERATURE=0.7
REPETITION_PENALTY=1.2

# Function to display usage
usage() {
    echo "Usage: $0 --prompt \"your text here\" [--model-name MODEL] [--topp VALUE] [--temperature VALUE] [--repetition-penalty VALUE]"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --prompt) PROMPT="$2"; shift ;;
        --model-name) MODEL_NAME="$2"; shift ;;
        --topp) TOPP="$2"; shift ;;
        --temperature) TEMPERATURE="$2"; shift ;;
        --repetition-penalty) REPETITION_PENALTY="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Ensure a prompt is provided
if [ -z "$PROMPT" ]; then
    echo "Error: --prompt argument is required."
    usage
fi

# Run the inference script
python benchmarks/benchmark_generation_mamba_simple.py \
    --model-name "$MODEL_NAME" \
    --prompt "$PROMPT" \
    --topp "$TOPP" \
    --temperature "$TEMPERATURE" \
    --repetition-penalty "$REPETITION_PENALTY"