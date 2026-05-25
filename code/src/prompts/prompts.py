# LLM_AS_A_JUDGE = """
# You are evaluating whether a response correctly answers a question.

# Question: "{question}"
# Response to evaluate: {system_generated_answer}
# Reference answer: {ground_truth_answer}

# Does the response convey the same facts as the reference? Focus on meaning, not wording.

# Score 1 if the response is factually equivalent — even if phrased differently or with extra detail, as long as nothing contradicts the reference.
# Score 0 if the response contradicts the reference, contains a factual error, or doesn't actually answer the question.

# Reply with only the number 0 or 1.
# """

LLM_AS_A_JUDGE = """
    ###Task Description:
    As an expert evaluator, compare the "Response to evaluate" against the "Reference Answer" for the question: "{question}"
    Determine if the response is factually equivalent to the reference.

    ###Evaluation Guidelines:
    - Focus on SEMANTIC equivalence, not exact wording or phrasing.
    - Paraphrases or different sentence structures that convey the same facts = Score 1.
    - Extra context or details that do not contradict the reference = Score 1.
    - The response must answer the question correctly to receive Score 1.
    - Only score 0 if the response contradicts the reference, contains a factual error, or does not answer the question.

    ###Few-Shot Examples:
    
    Question: "What is the boiling point of water?"
    Response: "Water boils at 100 degrees Celsius at sea level."
    Reference: "The boiling point of water is 100°C."
    Score: 1
    Reason: Same fact, extra context about sea level does not contradict the reference.

    Question: "Who wrote Romeo and Juliet?"
    Response: "Romeo and Juliet was authored by the famous English playwright William Shakespeare."
    Reference: "Romeo and Juliet was written by Shakespeare."
    Score: 1
    Reason: Semantically equivalent, additional detail about Shakespeare does not contradict the reference.

    Question: "What is the capital of France?"
    Response: "Lyon is the capital of France."
    Reference: "The capital of France is Paris."
    Score: 0
    Reason: Directly contradicts the reference — Lyon is not the capital of France.

    Question: "What is the speed of light?"
    Response: "The speed of light is approximately 300,000 km/s."
    Reference: "Light travels at exactly 299,792 km/s."
    Score: 0
    Reason: "Approximately 300,000" is a reasonable rounding, but paired with "exactly 299,792" in the reference this is a factual mismatch.

    Question: "What gas do plants absorb during photosynthesis?"
    Response: "During photosynthesis, plants take in carbon dioxide from the atmosphere and convert it into energy."
    Reference: "Plants absorb CO2 during photosynthesis."
    Score: 1
    Reason: Carbon dioxide and CO2 are the same substance, additional process detail does not contradict the reference.

    ###Constraints:
    Provide ONLY the numerical score.
    Do not provide any feedback, reasoning, or introductory text.

    ###Response to evaluate:
    {system_generated_answer}

    ###Reference Answer:
    {ground_truth_answer}

    ###Score Rubrics:
    [Factual Equivalence Alignment]
    Score 0: Incorrect. The response contradicts the reference, contains a factual error, or fails to answer the question.
    Score 1: Correct. The response is semantically equivalent to the reference and correctly answers the question.

    ###Result (Score Only):
"""

# LLM_AS_A_JUDGE = """
# ### Role:
# You are a precise, objective expert evaluator. Your goal is to determine if a "Response to evaluate" is factually equivalent to a "Reference Answer."

# ### Task:
# Compare the semantic meaning of the two texts below.
# 1. **Reference Answer**: {ground_truth_answer}
# 2. **Response to evaluate**: {system_generated_answer}
# 3. **Question Context**: {question}

# ### Evaluation Criteria:
# - Score 1 (Correct): The response conveys the same core facts as the reference. It can be a paraphrase, use different terminology (e.g., "CO2" vs "Carbon Dioxide"), or include extra non-contradictory info.
# - Score 0 (Incorrect): The response contains factual errors, directly contradicts the reference, or misses the core answer entirely.

# ### Evaluation Protocol:
# 1. **Analysis**: Break down the factual claims in both the reference and the response.
# 2. **Comparison**: Check for alignment or contradictions.
# 3. **Verdict**: Decide on a binary score (0 or 1).

# ### Result (Score Only):
# """