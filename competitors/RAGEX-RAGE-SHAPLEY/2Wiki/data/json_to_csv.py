import csv
import json

input_file_path = "dev.jsonl"
output_file_path = "2wiki_dev.csv"

CUSTOM_DELIMITER = "|||"
csv_headers = ["Question", "Golden Answer", "Supporting Paragraph"]

with open(input_file_path, "r", encoding="utf-8") as infile, open(
    output_file_path, "w", newline="", encoding="utf-8"
) as outfile:

    writer = csv.DictWriter(outfile, fieldnames=csv_headers, delimiter="|")
    writer.writeheader()

    for line in infile:
        if not line.strip():
            continue

        row = json.loads(line)

        question = row.get("question", "")

        golden_answers = row.get("golden_answers", [])
        final_answers_string = f" {CUSTOM_DELIMITER} ".join(
            [a.strip() for a in golden_answers if isinstance(a, str)]
        )


        metadata = row.get("metadata", {})
        context = metadata.get("context", {})

        titles = context.get("title", [])
        contents = context.get("content", [])

        title_to_content = {
            t: c for t, c in zip(titles, contents)
        }


        supporting_texts = []

        supporting_facts = metadata.get("supporting_facts", {})
        fact_titles = supporting_facts.get("title", [])
        sent_ids = supporting_facts.get("sent_id", [])

        for title, sent_id in zip(fact_titles, sent_ids):
            if title in title_to_content:
                paragraphs = title_to_content[title]

                if isinstance(paragraphs, list):
                    if 0 <= sent_id < len(paragraphs):
                        supporting_texts.append(paragraphs[sent_id].strip())


        combined_paragraphs = f" {CUSTOM_DELIMITER} ".join(supporting_texts)


        writer.writerow({
            "Question": question,
            "Golden Answer": final_answers_string,
            "Supporting Paragraph": combined_paragraphs,
        })

print(f"File created successfully at: {output_file_path}")