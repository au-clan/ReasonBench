#!/bin/bash

benchmark="game24"
method="io"
split="mini"
repeats=1

provider="together"
api_key="TOGETHER_API_KEY"
models=("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" "deepseek-ai/DeepSeek-R1" "openai/gpt-oss-120b" "Qwen/Qwen3-235B-A22B-Thinking-2507")

source scripts/configs/$benchmark.env

for model in "${models[@]}"; do
  echo "Running benchmark=$benchmark with model=$model"

  # Reset MAX_COMPLETION_TOKENS to default from env
  MAX_COMPLETION_TOKENS_DEFAULT="$MAX_COMPLETION_TOKENS"

  # Override for specific methods
  if [[ "$method" == "io" || "$method" == "cot" ]]; then
    MAX_COMPLETION_TOKENS=10000
  else
    MAX_COMPLETION_TOKENS="$MAX_COMPLETION_TOKENS_DEFAULT"
  fi

  python scripts/repeats/repeats.py \
      --benchmark "$benchmark" \
      --method "$method" \
      --model "$model" \
      --batch_size 1 \
      --timeout 2.0 \
      --temperature "$TEMPERATURE" \
      --max_completion_tokens "$MAX_COMPLETION_TOKENS" \
      --top_p "$TOP_P" \
      --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
      --split "$split" \
      --correctness 1 \
      --allow_batch_overflow 1 \
      --ns_ratio 0.0 \
      --provider "$provider" \
      --api_key "$api_key" \
      ${STOP:+--stop "$STOP"} \
      --value_cache \
      --repeats "$repeats"

  echo "Finished model=$model"
  echo "-----------------------------------"
done
