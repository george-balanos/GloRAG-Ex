## Implement Global Explanation Algorithm

### Aggregation Algorithm

Given a benchmark B and a list of n instances T = {q_1, ..., q_n}:
1. For each benchmark instance q_i, compute its local explanation using Algorithm 1.
2. Aggregate the n local explanations to discover which structural or semantic/attribute features influence tasks T.