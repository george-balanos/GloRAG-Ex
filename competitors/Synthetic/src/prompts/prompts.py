LLM_AS_A_JUDGE = """
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