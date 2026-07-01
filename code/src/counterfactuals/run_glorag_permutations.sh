python3 -m src.counterfactuals.permutation_robustness --dataset "synthetic" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/synthetic/delete_ops_ft --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_40_synthetic_ft.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/synthetic_ft_g_permutation.json

python3 -m src.counterfactuals.permutation_robustness --dataset "synthetic" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/synthetic/all_ops_ff --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_100_synthetic_ff.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/synthetic_ff_g_permutation.json



python3 -m src.counterfactuals.permutation_robustness --dataset "hotpotqa" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/hotpotqa/delete_ops_ft --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_50_hotpotqa_ft.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/hotpotqa_ft_g_permutation.json

python3 -m src.counterfactuals.permutation_robustness --dataset "hotpotqa" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/hotpotqa/all_ops_ff --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_60_hotpotqa_ff.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/hotpotqa_ff_g_permutation.json



python3 -m src.counterfactuals.permutation_robustness --dataset "musique" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/musique/delete_ops_ft --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_70_musique_ft.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/musique_ft_g_permutation.json

python3 -m src.counterfactuals.permutation_robustness --dataset "musique" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/musique/all_ops_ff --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_90_musique_ff.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/musique_ff_g_permutation.json


python3 -m src.counterfactuals.permutation_robustness --dataset "2wiki" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/2wiki/delete_ops_ft --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_45_2wiki_ft.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/2wiki_ft_g_permutation.json

python3 -m src.counterfactuals.permutation_robustness --dataset "2wiki" --input-dir /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/results/2wiki/all_ops_ff --questions-file /home/gbalanos/GloRAG-Ex/code/src/sampled_questions/sample_100_2wiki_ff.json --output /home/gbalanos/GloRAG-Ex/code/src/counterfactuals/permutations/2wiki_ff_g_permutation.json