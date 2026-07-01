import csv
import uuid

input_path = "/home/gbalanos/GloRAG-Ex/competitors/Musique/data/musique_train.csv"
output_questions = "/home/gbalanos/GloRAG-Ex/competitors/Musique/data/qa_data_musique.csv"
output_paragraphs = "/home/gbalanos/GloRAG-Ex/competitors/Musique/data/musique_supporting_facts.csv"

with open(input_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="|")
    rows = list(reader)

with open(output_questions, "w", encoding="utf-8", newline="") as qf, \
     open(output_paragraphs, "w", encoding="utf-8", newline="") as pf:

    q_writer = csv.DictWriter(qf, fieldnames=["id", "question", "answer"])
    p_writer = csv.DictWriter(pf, fieldnames=["id", "paragraphs"])

    q_writer.writeheader()
    p_writer.writeheader()

    for row in rows:
        row_id = uuid.uuid4().hex[:24]

        answers = [a.strip() for a in row["Golden Answer"].split("|||")]
        longest_answer = max(answers, key=len)

        q_writer.writerow({
            "id": row_id,
            "question": row["Question"].strip(),
            "answer": longest_answer
        })

        paragraphs = [p.strip() for p in row["Supporting Paragraph"].split("|||")]
        p_writer.writerow({
            "id": row_id,
            "paragraphs": " ||| ".join(paragraphs)
        })

print(f"Converted {len(rows)} rows.")
print(f"Questions saved to {output_questions}")
print(f"Paragraphs saved to {output_paragraphs}")