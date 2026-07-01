#!/bin/bash
set -e  # stop if either command fails

python3 -m src.counterfactuals.generate --dataset "musique" \
    --mode "ft" --rag-mode "hybrid" --top-k 2 --ops delete_node,delete_edge \
    --max-cost 20 --max-llm-calls 200 --adm 2

python3 -m src.counterfactuals.generate --dataset "musique" \
    --mode "ff" --rag-mode "hybrid" --top-k 2 --ops add_node,add_edge,delete_node,delete_edge \
    --max-cost 20 --max-llm-calls 200 --adm 2