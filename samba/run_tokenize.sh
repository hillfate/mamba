# python scripts/prepare_slimpajama.py \
#     --source_path /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/SlimPajama-627B \
#     --tokenizer_path /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/tokenizer_llama2 \
#     --destination_path /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/tokenized_data \
#     --split validation --percentage 1.0


python scripts/prepare_slimpajama.py \
    --source_path /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/SlimPajama-627B \
    --tokenizer_path /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/tokenizer_llama2 \
    --destination_path /network/rit/dgx/dgx_yinlab/ydeng/data/slimPajama/tokenized_data \
    --split train --percentage 1.0

