python3 /home/gbalanos/GloRAG-Ex/competitors/Shapley/run_shapley_noise.py --dataset "synthetic" --rag-mode "hybrid" --top-k 2 --noise-percentages "0.1,0.2,0.3,0.5" --top-attr-ks "1,2,3,4,5" --judge --verbose --output "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/synthetic_shapley_noise.json" --metrics "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/synthetic_shapley_noise_metrics.json" --num-rows 100


python3 /home/gbalanos/GloRAG-Ex/competitors/Shapley/run_shapley_noise.py --dataset "hotpotqa" --rag-mode "hybrid" --top-k 2 --noise-percentages "0.1,0.2,0.3,0.5" --top-attr-ks "1,2,3,4,5" --judge --verbose --output "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/hotpotqa_shapley_noise.json" --metrics "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/hotpotqa_shapley_noise_metrics.json" --num-rows 200


python3 /home/gbalanos/GloRAG-Ex/competitors/Shapley/run_shapley_noise.py --dataset "musique" --rag-mode "hybrid" --top-k 2 --noise-percentages "0.1,0.2,0.3,0.5" --top-attr-ks "1,2,3,4,5" --judge --verbose --output "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/musique_shapley_noise.json" --metrics "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/musique_shapley_noise_metrics.json" --num-rows 200


python3 /home/gbalanos/GloRAG-Ex/competitors/Shapley/run_shapley_noise.py --dataset "2wiki" --rag-mode "hybrid" --top-k 2 --noise-percentages "0.1,0.2,0.3,0.5" --top-attr-ks "1,2,3,4,5" --judge --verbose --output "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/2wiki_shapley_noise.json" --metrics "/home/gbalanos/GloRAG-Ex/competitors/Shapley/robustness/2wiki_shapley_noise_metrics.json" --num-rows 300