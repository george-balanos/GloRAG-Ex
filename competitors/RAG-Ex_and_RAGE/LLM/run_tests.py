from LLM.LLMWrapper import LLMWrapper

if __name__ == "__main__":
    lw = LLMWrapper()
    # lw.evaluate_file("master_synthetic_dataset.csv", "llm_only")
    # lw.evaluate_file("master_synthetic_dataset.csv", "rag")
    lw.compare_answers("master_synthetic_dataset", "remove_sentence")
    lw.extract_impactful_changes("master_synthetic_dataset", "remove_sentence")

    lw.compare_answers("master_synthetic_dataset", "rage")
    lw.extract_impactful_changes("master_synthetic_dataset", "rage")

    lw.compare_answers("master_synthetic_dataset", "remove_word")
    lw.extract_impactful_changes("master_synthetic_dataset", "remove_word")