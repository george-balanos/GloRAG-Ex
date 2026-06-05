from datetime import datetime
from src.query import *
from src.retrieve import *
from src.parser import *
from src.llm_judge import judge_response
from .perturbations import delete_node, delete_edge
from src.counterfactuals.utils import compute_answer_similarity, cosine_similarity_norm
from .edit_costs import delete_edge_cost, delete_node_cost

import networkx as nx
import asyncio
import itertools
import os


### Setup ###

dataset = "synthetic"
G = nx.read_graphml(f"KGs/lightrag/{dataset}/graph_chunk_entity_relation.graphml")

##################################################

async def find_breaking_nodes_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    max_llm_calls: int = 100,
    current_ops: list = ["delete_node", "delete_edge"],
    mode: str = "ft"      
):
    
    llm_calls = 0

    ### Lightrag specific ###
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #########################
    
    context_graph_nodes = set(context_graph.nodes)

    llm_calls, found = await apply_breaking_node_perturbations(
        rag,
        question=question,
        original_answer=original_answer,
        ground_truth=ground_truth,
        context=context,
        context_graph_nodes=context_graph_nodes,
        context_graph=context_graph,
        max_llm_calls=max_llm_calls,
        current_ops=current_ops,
        mode=mode,
        llm_calls=llm_calls
    )

    if found == False:
        print(f"Could not find feasible counterfactual explanations.")

        save_operations_to_json(
            ops=[],
            question=question,
            ground_truth=ground_truth,
            original_answer=original_answer,
            perturbed_answer=None,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=None,
            found=False,
            llm_calls=llm_calls,
            current_ops=current_ops,
            mode=mode,
            output_dir=f"KGRAG-Ex/results/{dataset}/node"
        )

async def find_breaking_edges_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    max_llm_calls: int = 100,
    current_ops: list = ["delete_node", "delete_edge"],
    mode: str = "ft"      
):
    
    llm_calls = 0

    ### Lightrag specific ###
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #########################
    
    context_graph_edges = set(context_graph.edges())

    llm_calls, found = await apply_breaking_edge_perturbations(
        rag,
        question=question,
        original_answer=original_answer,
        ground_truth=ground_truth,
        context=context,
        context_graph_edges=context_graph_edges,
        context_graph=context_graph,
        max_llm_calls=max_llm_calls,
        current_ops=current_ops,
        mode=mode,
        llm_calls=llm_calls
    )

    if found == False:
        print(f"Could not find feasible counterfactual explanations.")

        save_operations_to_json(
            ops=[],
            question=question,
            ground_truth=ground_truth,
            original_answer=original_answer,
            perturbed_answer=None,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=None,
            found=False,
            llm_calls=llm_calls,
            current_ops=current_ops,
            mode=mode,
            output_dir=f"KGRAG-Ex/results/{dataset}/edge"
        )

async def apply_breaking_node_perturbations(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    context_graph_nodes: list,
    context_graph: nx.DiGraph,
    max_llm_calls: int,
    current_ops: list,
    mode: str,
    llm_calls: int
):
    for node in context_graph_nodes:
        if llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        ops = [node]

        llm_calls += 1
        
        perturbed_cg = delete_node(context_graph, node)
        perturbation_cost = delete_node_cost(context_graph, node)

        cg_context = graph_to_context(perturbed_cg)
        new_response = await query(rag, cg_context, question)

        print(f"New response: {new_response} | Original response: {original_answer}")
        print(f"Ground Truth: {ground_truth}")

        score = await judge_response(question, new_response, original_answer)
        if score == 0:
            print(f"Counterfactual Operations: {ops}")

            answer_similarity = await compute_answer_similarity(original_answer, new_response)
            print(f"Answer similarity (original vs perturbed): {answer_similarity:.4f}")

            parsed_subgraph = parse_context(context)

            save_operations_to_json(
                ops=ops,
                question=question,
                ground_truth=ground_truth,
                original_answer=original_answer,
                perturbed_answer=new_response,
                answer_similarity=answer_similarity,
                original_subgraph=parsed_subgraph,
                perturbed_subgraph=graph_to_subgraph(perturbed_cg),
                found=True,
                llm_calls=llm_calls,
                current_ops=current_ops,
                mode=mode,
                output_dir=f"KGRAG-Ex/results/{dataset}/node",
                cost=perturbation_cost
            )
            
            return llm_calls, True

    return llm_calls, False

async def apply_breaking_edge_perturbations(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    context_graph_edges: list,
    context_graph: nx.DiGraph,
    max_llm_calls: int,
    current_ops: list,
    mode: str,
    llm_calls: int
):
    for edge in context_graph_edges:
        if llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        ops = [edge]

        llm_calls += 1
        
        perturbed_cg = delete_edge(context_graph, edge)
        perturbation_cost = delete_edge_cost(context_graph, edge)

        cg_context = graph_to_context(perturbed_cg)
        new_response = await query(rag, cg_context, question)

        print(f"New response: {new_response} | Original response: {original_answer}")
        print(f"Ground Truth: {ground_truth}")

        score = await judge_response(question, new_response, original_answer)
        if score == 0:
            print(f"Counterfactual Operations: {ops}")

            answer_similarity = await compute_answer_similarity(original_answer, new_response)
            print(f"Answer similarity (original vs perturbed): {answer_similarity:.4f}")

            parsed_subgraph = parse_context(context)

            save_operations_to_json(
                ops=ops,
                question=question,
                ground_truth=ground_truth,
                original_answer=original_answer,
                perturbed_answer=new_response,
                answer_similarity=answer_similarity,
                original_subgraph=parsed_subgraph,
                perturbed_subgraph=graph_to_subgraph(perturbed_cg),
                found=True,
                llm_calls=llm_calls,
                current_ops=current_ops,
                mode=mode,
                output_dir=f"KGRAG-Ex/results/{dataset}/edge",
                cost=perturbation_cost
            )

            return llm_calls, True
        
    return llm_calls, False

##################################################

async def find_corrective_nodes_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    max_llm_calls: int = 100,
    current_ops: list = ["delete_node", "delete_edge"],
    mode: str = "ff"      
):
    
    llm_calls = 0

    ### Lightrag specific ###
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #########################
    
    context_graph_nodes = set(context_graph.nodes)

    llm_calls, found = await apply_corrective_node_perturbations(
        rag,
        question=question,
        original_answer=original_answer,
        ground_truth=ground_truth,
        context=context,
        context_graph_nodes=context_graph_nodes,
        context_graph=context_graph,
        max_llm_calls=max_llm_calls,
        current_ops=current_ops,
        mode=mode,
        llm_calls=llm_calls
    )

    if found == False:
        print(f"Could not find feasible counterfactual explanations.")

        save_operations_to_json(
            ops=[],
            question=question,
            ground_truth=ground_truth,
            original_answer=original_answer,
            perturbed_answer=None,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=None,
            found=False,
            llm_calls=llm_calls,
            current_ops=current_ops,
            mode=mode,
            output_dir=f"KGRAG-Ex/results/{dataset}/node"
        )

async def find_corrective_edges_counterfactuals(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    max_llm_calls: int = 100,
    current_ops: list = ["delete_node", "delete_edge"],
    mode: str = "ft"      
):
    
    llm_calls = 0

    ### Lightrag specific ###
    parsed_subgraph = parse_context(context)
    context_graph = parse_graph(parsed_subgraph)
    #########################
    
    context_graph_edges = set(context_graph.edges())

    llm_calls, found = await apply_corrective_edge_perturbations(
        rag,
        question=question,
        original_answer=original_answer,
        ground_truth=ground_truth,
        context=context,
        context_graph_edges=context_graph_edges,
        context_graph=context_graph,
        max_llm_calls=max_llm_calls,
        current_ops=current_ops,
        mode=mode,
        llm_calls=llm_calls
    )

    if found == False:
        print(f"Could not find feasible counterfactual explanations.")

        save_operations_to_json(
            ops=[],
            question=question,
            ground_truth=ground_truth,
            original_answer=original_answer,
            perturbed_answer=None,
            answer_similarity=0.0,
            original_subgraph=parsed_subgraph,
            perturbed_subgraph=None,
            found=False,
            llm_calls=llm_calls,
            current_ops=current_ops,
            mode=mode,
            output_dir=f"KGRAG-Ex/results/{dataset}/edge"
        )

async def apply_corrective_node_perturbations(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    context_graph_nodes: list,
    context_graph: nx.DiGraph,
    max_llm_calls: int,
    current_ops: list,
    mode: str,
    llm_calls: int
):
    for node in context_graph_nodes:
        if llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        ops = [node]

        llm_calls += 1
        
        perturbed_cg = delete_node(context_graph, node)
        perturbation_cost = delete_node_cost(context_graph, node)


        cg_context = graph_to_context(perturbed_cg)
        new_response = await query(rag, cg_context, question)

        print(f"New response: {new_response} | Original response: {original_answer}")
        print(f"Ground Truth: {ground_truth}")

        score = await judge_response(question, new_response, ground_truth)
        if score == 1:
            print(f"Counterfactual Operations: {ops}")

            answer_similarity = await compute_answer_similarity(ground_truth, new_response)
            print(f"Answer similarity (ground truth vs perturbed): {answer_similarity:.4f}")

            parsed_subgraph = parse_context(context)

            save_operations_to_json(
                ops=ops,
                question=question,
                ground_truth=ground_truth,
                original_answer=original_answer,
                perturbed_answer=new_response,
                answer_similarity=answer_similarity,
                original_subgraph=parsed_subgraph,
                perturbed_subgraph=graph_to_subgraph(perturbed_cg),
                found=True,
                llm_calls=llm_calls,
                current_ops=current_ops,
                mode=mode,
                output_dir=f"KGRAG-Ex/results/{dataset}/node",
                cost=perturbation_cost
            )
            
            return llm_calls, True

    return llm_calls, False

async def apply_corrective_edge_perturbations(
    rag,
    question: str,
    original_answer: str,
    ground_truth: str,
    context: str,
    context_graph_edges: list,
    context_graph: nx.DiGraph,
    max_llm_calls: int,
    current_ops: list,
    mode: str,
    llm_calls: int
):
    for edge in context_graph_edges:
        if llm_calls > max_llm_calls:
            print(f"Max LLM calls {max_llm_calls} exceeded. Stopping search.")
            break

        ops = [edge]

        llm_calls += 1
        
        perturbed_cg = delete_edge(context_graph, edge)
        perturbation_cost = delete_edge_cost(context_graph, edge)

        cg_context = graph_to_context(perturbed_cg)
        new_response = await query(rag, cg_context, question)

        print(f"New response: {new_response} | Original response: {original_answer}")
        print(f"Ground Truth: {ground_truth}")

        score = await judge_response(question, new_response, ground_truth)
        if score == 1:
            print(f"Counterfactual Operations: {ops}")

            answer_similarity = await compute_answer_similarity(ground_truth, new_response)
            print(f"Answer similarity (ground truth vs perturbed): {answer_similarity:.4f}")

            parsed_subgraph = parse_context(context)

            save_operations_to_json(
                ops=ops,
                question=question,
                ground_truth=ground_truth,
                original_answer=original_answer,
                perturbed_answer=new_response,
                answer_similarity=answer_similarity,
                original_subgraph=parsed_subgraph,
                perturbed_subgraph=graph_to_subgraph(perturbed_cg),
                found=True,
                llm_calls=llm_calls,
                current_ops=current_ops,
                mode=mode,
                output_dir=f"KGRAG-Ex/results/{dataset}/edge",
                cost=perturbation_cost
            )
            
            return llm_calls, True

    return llm_calls, False


async def find_counterfactuals(
    rag, 
    question: str, 
    context,  
    max_llm_calls=100, 
    current_ops: list=["delete_node", "delete_edge", "replace_node", "replace_edge"], 
    ground_truth: str = "",
    mode: str = "ft"
):
    original_answer = await query(rag, context, question)

    if mode == "ft":
        await find_breaking_nodes_counterfactuals(
            rag=rag,
            question=question,
            original_answer=original_answer,
            ground_truth=ground_truth,
            context=context,
            max_llm_calls=max_llm_calls,
            current_ops=current_ops,
            mode=mode
        )

        await find_breaking_edges_counterfactuals(
            rag=rag,
            question=question,
            original_answer=original_answer,
            ground_truth=ground_truth,
            context=context,
            max_llm_calls=max_llm_calls,
            current_ops=current_ops,
            mode=mode
        )
    elif mode == "ff":
        await find_corrective_nodes_counterfactuals(
            rag=rag,
            question=question,
            original_answer=original_answer,
            ground_truth=ground_truth,
            context=context,
            max_llm_calls=max_llm_calls,
            current_ops=current_ops,
            mode=mode
        )

        await find_corrective_edges_counterfactuals(
            rag=rag,
            question=question,
            original_answer=original_answer,
            ground_truth=ground_truth,
            context=context,
            max_llm_calls=max_llm_calls,
            current_ops=current_ops,
            mode=mode
        )

        
def save_operations_to_json(
    ops: list, 
    question: str, 
    ground_truth: str,
    original_answer: str, 
    perturbed_answer: str, 
    answer_similarity: float, 
    original_subgraph, 
    perturbed_subgraph, 
    output_dir: str = "KGRAG-Ex/results", 
    filename: str = None, 
    found: bool = True, 
    llm_calls: int = 0, 
    current_ops: list=[],
    mode: str = "ff",
    cost: float = 0.0,
):

    if current_ops == ["delete_node", "delete_edge"]:
        output_dir = f"{output_dir}/delete_ops_{mode}"

    os.makedirs(output_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"counterfactual_{timestamp}.json"

    filepath = os.path.join(output_dir, filename)

    serialisable_ops = []
    for op in ops:
        if isinstance(op, tuple):
            serialisable_ops.append(list(op))
        else:
            serialisable_ops.append(op)

    payload = {
        "question": question,
        "found": found,
        "num_operations": len(serialisable_ops),
        "operations": serialisable_ops,
        "cost": cost,
        "answers": {
            "ground_truth": ground_truth,
            "original": original_answer,
            "perturbed": perturbed_answer,
            "similarity": round(answer_similarity, 6)
        },
        "original_subgraph": subgraph_to_dict(original_subgraph),
        "perturbed_subgraph": subgraph_to_dict(perturbed_subgraph),
        "timestamp": datetime.now().isoformat(),
        "llm_calls": llm_calls,
        "mode": mode
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Operations saved to: {filepath}")
    return filepath


async def main():
    global mode

    rag = await initialize_lightrag(working_dir=WORKING_DIR_SYNTHETIC)
    
    with open(f"benchmark/results/comparison_{dataset}_2.json", "r", encoding="utf-8") as results:
        data = json.load(results)

    operation_sets = [
        ["delete_node", "delete_edge"]
    ]

    mode = "ft"

    for op_set in operation_sets:
        results = data["results"]
        for idx, r in results.items():
            question = r["question"]
            case = r["case"]

            ground_truth = r["ground_truth"]

            if case != mode:
                continue

            print(f"\n=== [{idx}] {question} ===")

            context = await retrieve_subgraph(
                rag, 
                query=question, 
                mode="hybrid", 
                top_k=2
            )

            await find_counterfactuals(
                rag=rag, 
                question=question, 
                context=context,  
                max_llm_calls=200, 
                current_ops=op_set, 
                ground_truth=ground_truth,
                mode=mode
            )

if __name__ == "__main__":
    asyncio.run(main())