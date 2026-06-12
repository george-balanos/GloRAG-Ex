import csv
import json

# Name of your input file containing the JSON text line by line
input_file_path = "train.jsonl"
output_file_path = "musique_train.csv"
# Our custom list separator token
CUSTOM_DELIMITER = "|||"

csv_headers = ["Question", "Golden Answer", "Supporting Paragraph"]

# FIXED: Changed 'custom_csv_path' to 'output_file_path'
with open(input_file_path, "r", encoding="utf-8") as infile, open(
    output_file_path, "w", newline="", encoding="utf-8"
) as outfile:

    writer = csv.DictWriter(
        outfile, fieldnames=csv_headers, delimiter="|"
    )
    writer.writeheader()

    for line in infile:
        if not line.strip():
            continue

        row_data = json.loads(line)
        question = row_data.get("question", "")

        # 1. Join Golden Answers using the custom delimiter
        raw_answers = row_data.get("golden_answers", [])
        final_answers_string = f" {CUSTOM_DELIMITER} ".join(
            [ans.strip() for ans in raw_answers]
        )

        # 2. Extract Supporting Paragraphs
        supporting_texts = []
        decompositions = row_data.get("metadata", {}).get(
            "question_decomposition", []
        )
        for decomp in decompositions:
            support_block = decomp.get("support_paragraph", {})
            if support_block.get("is_supporting") is True:
                para_text = support_block.get("paragraph_text")
                if para_text:
                    supporting_texts.append(para_text.strip())

        # Join Paragraphs using the EXACT same custom delimiter
        combined_paragraphs = f" {CUSTOM_DELIMITER} ".join(supporting_texts)

        # Write to our CSV file
        writer.writerow(
            {
                "Question": question,
                "Golden Answer": final_answers_string,
                "Supporting Paragraph": combined_paragraphs,
            }
        )

# FIXED: Changed 'custom_csv_path' to 'output_file_path'
print(f"File created successfully at: {output_file_path}")