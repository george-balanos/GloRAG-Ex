#!/bin/bash
set -e  # stop if either command fails

python3 -m src.quality_metrics.noise_resistance \
    --dataset musique --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/musique/delete_ops_ft --mode ft --ops "delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.1,0.2,0.3,0.5" --output-dir src/counterfactuals/robustness/musique/noise_resistance

python3 -m src.quality_metrics.noise_resistance \
    --dataset musique --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/musique/all_ops_ff --mode ff --ops "add_node,add_edge,delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.5" --output-dir src/counterfactuals/robustness/musique/noise_resistance



python3 -m src.quality_metrics.noise_resistance \
    --dataset synthetic --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/synthetic/delete_ops_ft --mode ft --ops "delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.1,0.2,0.3,0.5" --output-dir src/counterfactuals/robustness/synthetic/noise_resistance

python3 -m src.quality_metrics.noise_resistance \
    --dataset synthetic --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/synthetic/all_ops_ff --mode ff --ops "add_node,add_edge,delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.1,0.2,0.3,0.5" --output-dir src/counterfactuals/robustness/synthetic/noise_resistance



python3 -m src.quality_metrics.noise_resistance \
    --dataset hotpotqa --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/hotpotqa/delete_ops_ft --mode ft --ops "delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.1,0.2,0.3,0.5" --output-dir src/counterfactuals/robustness/hotpotqa/noise_resistance

python3 -m src.quality_metrics.noise_resistance \
    --dataset hotpotqa --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/hotpotqa/all_ops_ff --mode ff --ops "add_node,add_edge,delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.1,0.2,0.3,0.5" --output-dir src/counterfactuals/robustness/hotpotqa/noise_resistance



python3 -m src.quality_metrics.noise_resistance \
    --dataset 2wiki --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/2wiki/delete_ops_ft --mode ft --ops "delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.1,0.2,0.3,0.5" --output-dir src/counterfactuals/robustness/2wiki/noise_resistance

python3 -m src.quality_metrics.noise_resistance \
    --dataset 2wiki --rag-mode hybrid --top-k 2 \
    --input src/counterfactuals/results/2wiki/all_ops_ff --mode ff --ops "add_node,add_edge,delete_node,delete_edge" \
    --max-cost 20 --max-llm-calls 200 --noise-percentages "0.1,0.2,0.3,0.5" --output-dir src/counterfactuals/robustness/2wiki/noise_resistance