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
    - If the reference answer is a short/atomic answer (e.g. a name, number, or label), the response scores 1 if it correctly contains or states that answer, even with additional elaboration — as long as nothing contradicts it.
    - A response that refuses to answer, claims ignorance, or says "I don't know", "I don't have enough information" = Score 0, regardless of whether it contradicts the reference. Failing to answer is always incorrect.

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




################## Medical #####################

EXTRACT_MEDICAL_ENTITIES = """*** Role ***
You are an expert medical NLP system responsible for extracting biomedical entities and patient demographics from clinical and scientific text.

*** Goal ***
Extract all medically significant entities from the input text. Every entity must belong to one of the following categories:

1)  Disease          - e.g. Type 2 Diabetes Mellitus, Crohn's Disease, myocardial infarction
2)  Symptom          - e.g. dyspnea, tachycardia, chest pain, tinnitus
3)  Drug             - e.g. cisplatin, metformin, atorvastatin (use INN names where possible)
4)  Anatomy          - e.g. left ventricle, cochlear hair cells, femoral artery
5)  Treatment        - e.g. coronary angioplasty, neoadjuvant chemotherapy, radiation therapy
6)  Complication     - e.g. ototoxicity, nephrotoxicity, sensorineural hearing loss
7)  Etiology         - e.g. reactive oxygen species, DNA cross-linking, cholesterol embolization
8)  Patient_Pop      - e.g. 67-year-old man, pediatric patients, pregnant women
9)  Professional_Role - e.g. attending physician, orthopaedic resident, radiologist
10) Diagnostic       - e.g. pure tone audiometry, CBC, serum troponin, CT scan

*** Instructions ***
- Extract ANY clinically or scientifically relevant entity from the text.
- Every entity MUST include the "entity_category" field. Omitting it will render the output invalid.
- Prefer standardized nomenclature: INN drug names, standard anatomical terms, OMIM disease names.
- Do NOT extract relationships, explanations, or non-entity phrases.
- Do NOT include any text outside the JSON structure.

*** Output Format ***
Return JSON ONLY in this exact structure:

{{
  "entities": [
    {{
      "entity_category": "<one of the 10 categories above>",
      "entity_name": "<standardized entity name>"
    }}
  ]
}}

*** Example ***

Input:
A 67-year-old man with transitional cell carcinoma of the bladder presents with tinnitus one week after his first course of neoadjuvant cisplatin-based chemotherapy. Pure tone audiometry reveals sensorineural hearing loss of 45 dB.

Output:
{{
  "entities": [
    {{"entity_category": "Patient_Pop",  "entity_name": "67-year-old man"}},
    {{"entity_category": "Disease",      "entity_name": "Transitional Cell Carcinoma"}},
    {{"entity_category": "Symptom",      "entity_name": "Tinnitus"}},
    {{"entity_category": "Drug",         "entity_name": "Cisplatin"}},
    {{"entity_category": "Treatment",    "entity_name": "Neoadjuvant Chemotherapy"}},
    {{"entity_category": "Diagnostic",   "entity_name": "Pure Tone Audiometry"}},
    {{"entity_category": "Complication", "entity_name": "Sensorineural Hearing Loss"}}
  ]
}}

*** Input ***

{input_text}

*** Response ***
"""

MEDICAL_RAG_PROMPT = """*** Role ***
You are a helpful RAG (Retrieval-Augmented Generation) assistant that answers multiple-choice biomedical questions using the provided context.

*** Goal ***
Select the option that best answers the biomedical question, using the provided context as your primary source of reasoning.

*** Instructions ***
1) Respond ONLY in JSON format.
2) Return only the letter corresponding to your chosen answer: 'A', 'B', 'C', or 'D'.
3) Do NOT include any explanations, reasoning, or additional text outside the JSON.

*** Input ***

## Question:
{input_question}

## Options:
{options}

## Context:
{context}

*** Response ***"""


MEDICAL_LLM_ONLY_PROMPT = """*** Role ***
You are a helpful assistant that answers multiple-choice biomedical questions.

*** Goal ***
Select the option that best answers the biomedical question based on your medical knowledge.

*** Instructions ***
1) Respond ONLY in JSON format.
2) Return only the letter corresponding to your chosen answer: 'A', 'B', 'C', or 'D'.
3) Do NOT include any explanations, reasoning, or additional text outside the JSON.

*** Input ***

## Question:
{input_question}

## Options:
{options}

*** Response ***"""