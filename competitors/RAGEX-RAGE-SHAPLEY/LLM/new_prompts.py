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
    - All distinct facts or conditions in the reference must be addressed. A response that answers only part of a multi-part question = Score 0.

    ###Few-Shot Examples:

    Question: "Who invented the telephone?"
    Response: "The telephone was invented by Alexander Graham Bell, a Scottish-American inventor who patented the device in 1876."
    Reference: "Alexander Graham Bell"
    Score: 1
    Reason: The response directly names Alexander Graham Bell and adds supporting context that does not contradict the reference.

    Question: "Which country has the larger population, Canada or Brazil?"
    Response: "Brazil has a significantly larger population than Canada, with over 200 million people compared to Canada's 38 million."
    Reference: "Brazil"
    Score: 1
    Reason: The response correctly identifies Brazil as the answer with elaborating detail; the single-word reference is fully satisfied.

    Question: "Which artist won the award?"
    Response: "The award was given to Maria, who had competed against Jonas for years."
    Reference: "Jonas"
    Score: 0
    Reason: The response names Maria as the winner, which directly contradicts the reference answer of Jonas.

    Question: "What are the special traits of a Blorkin and a Snazzlewump?"
    Response: "The Blorkin can phase through solid matter using Vortex energy."
    Reference: "The Blorkin can phase through solid matter using Vortex energy, and the Snazzlewump secretes Luminal dust that renders it invisible."
    Score: 0
    Reason: The response only addresses the Blorkin. The Snazzlewump's trait is a required condition in the reference and is entirely missing.

    Question: "What is the capital of Australia?"
    Response: "I don't have enough information to answer that question."
    Reference: "Canberra"
    Score: 0
    Reason: The response does not answer the question at all. A refusal or claim of ignorance is always incorrect, even if it does not contradict the reference.
    
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

RAG_PROMPT = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Knowledge Graph Data` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""