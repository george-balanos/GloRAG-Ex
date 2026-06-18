from src.medical.retriever import *
from src.medical.query import *
from src.medical.extract_entities import *
from tqdm import tqdm
from src.medical.parser import graph_to_context

import pandas as pd
import asyncio
import json
import logging

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("nano-vectordb").setLevel(logging.WARNING)

def load_partition(benchmark_path: str, partition: str) -> pd.DataFrame:
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)

    if partition.startswith("mmlu_"):
        subcat = partition[len("mmlu_"):]   # e.g. "anatomy"
        top_cat = "mmlu"
        items = {
            k: v for k, v in data[top_cat].items()
            if k.rsplit("-", 1)[0] == subcat
        }
    else:
        top_cat = partition
        items = data[top_cat]

    rows = []
    for key, item in items.items():
        answer_letter = item["answer"]
        answer_text   = item["options"].get(answer_letter, answer_letter)
        rows.append({
            "id":        f"{top_cat}-{key}",
            "questions": item["question"],
            "options":   " | ".join(f"{k}: {v}" for k, v in item["options"].items()),
            "answers":   f"{answer_letter}",
        })

    return pd.DataFrame(rows, columns=["id", "questions", "options", "answers"])

async def run_example(question: str, options: str, ground_truth: str, depth=1):
    entities = await extract_entities(input_text=question)
    print(entities)

    validated_entities = validate_entity(G, entities)
    found_entities = validated_entities["found"]
    not_found_entities = validated_entities["not_found"]
    
    seed_nodes = found_entities
    for ent in not_found_entities:
        most_similar_node_id = find_similar_node_id(index=index, records=records, entity=ent)
        if most_similar_node_id:
            seed_nodes.append(most_similar_node_id)   

    subgraph: nx.DiGraph = bfs_subgraph(G, seed_nodes=seed_nodes, depth=depth) # BFS-Depth-1
    # subgraph: nx.DiGraph = shortest_paths_subgraph(G, seed_nodes=seed_nodes)
    subgraph = prune_subgraph(subgraph, query_text=question, top_k_nodes=5, top_k_edges=5, lookup=lookup, embeddings=embeddings)  # ← prune before context
    context_graph = graph_to_context(subgraph)

    answer = await query_rag(
        input_question=question,
        options=options,
        context=context_graph
    )

    answer = answer.upper()
    if answer in ["A", "B", "C", "D"]:
        score = 1 if (answer == ground_truth) else 0
        return score, answer
    
async def run_example_llm_only(question: str, options: str, ground_truth: str):
    answer = await query_llm_only(
        input_question=question,
        options=options,
    )

    answer = answer.upper()
    if answer in ["A", "B", "C", "D"]:
        score = 1 if (answer == ground_truth) else 0
        return score, answer
    
async def run_benchmark(mode="llm_only", num_rows=100, partition="mmlu", depth=2):
    # benchmark_data = load_partition("datasets/medical/benchmark.json", partition)
    benchmark_data = load_partition(f"datasets/medical/{partition}_ff.json", partition)

    if num_rows is not None:
        benchmark_data = benchmark_data.head(num_rows)

    result_dict = {}

    for i, row in tqdm(benchmark_data.iterrows(), desc="Processing questions...", total=len(benchmark_data)):
        id = row["id"]
        question = row["questions"]
        answer = row["answers"]
        options = str(row["options"])

        if mode == "llm_only":
            score, generated_answer = await run_example_llm_only(question, options, answer)
        else:
            score, generated_answer = await run_example(question, options, answer, depth=depth)

        print(f"Score: {score}\nGenerated Answer: {generated_answer} VS Ground Truth: {answer}")
        
        result_dict[id] = {
            "score": score,
            "generated_answer": generated_answer,
            "question": question,
            "ground_truth": answer
        }

    if mode == "llm_only":
        output_path = f"benchmark/results/medical_{mode}_{partition}.json"
    else:
        output_path = f"benchmark/results/medical_{mode}_{partition}_from_ff_{depth}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

async def main():
    mode = "rag" # "llm_only" or "rag"
    depth = 1 # 0 => Shortest-path

    partitions = ["bioasq"]

    for p in partitions:
        await run_benchmark(mode, depth=depth, partition=p, num_rows=None)

if __name__ == "__main__":
    G = nx.read_graphml(f"/home/gbalanos/GloRAG-Ex/code/KGs/medical/graph_chunk_entity_relation.graphml")

    index_prefix = "src/embeddings/medical/node_index"
    index, records, embeddings = load_index(index_prefix, DIM, 2000)
    lookup = build_lookup(records)

    asyncio.run(main())

# # top-level partitions
# df = load_partition("datasets/medical/benchmark.json", "medqa")
# df = load_partition("datasets/medical/benchmark.json", "bioasq")

# # mmlu subcategories
# df = load_partition("datasets/medical/benchmark.json", "mmlu_anatomy")
# df = load_partition("datasets/medical/benchmark.json", "mmlu_professional_medicine")