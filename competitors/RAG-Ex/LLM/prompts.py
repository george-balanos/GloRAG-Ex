LLM_AS_A_JUDGE_PROMPT = """
    ###Task Description:
    As an expert evaluator, compare the "Response to evaluate" against the "Reference Answer" for the question: "{question}"
    Determine if the response is factually equivalent to the reference. 

    ###Constraints:
    Provide ONLY the numerical score.
    Do not provide any feedback, reasoning, or introductory text.

    ###Response to evaluate:
    {system_generated_answer}

    ###Reference Answer:
    {ground_truth_answer}

    ###Score Rubrics:
    [Factual Equivalence Alignment]
    Score 0: Incorrect. The response contains a factual error or contradicts the reference.
    Score 1: Correct. The response is factually equivalent to the reference.

    ###Result (Score Only):

"""


QA_PROMPT = """
You are a helpful assistant.

Your job is to answer the question using ONLY the provided context.

If the context does not contain enough information, say:
"I don't know based on the provided context."

---

Context:
{context}

Question:
{question}

---

Answer:
Return only the final answer. Do not explain your reasoning.
"""