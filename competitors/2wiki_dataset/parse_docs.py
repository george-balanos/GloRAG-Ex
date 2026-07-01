import json

with open("/home/gbalanos/GloRAG-Ex/competitors/2wiki_dataset/2wiki_dev.jsonl", "r") as f:
    for line in f:
        row = json.loads(line)
        
        metadata = row["metadata"]
        context = metadata["context"]
        supporting_facts = metadata["supporting_facts"]
        
        titles = context["title"]
        contents = context["content"]
        title_to_content = dict(zip(titles, contents))
        
        # Get only the relevant titles (not distractors)
        relevant_titles = set(supporting_facts["title"])
        
        # Build full paragraph per relevant article
        for title in relevant_titles:
            if title in title_to_content:
                full_paragraph = " ".join(title_to_content[title])
                print(f"[{title}]\n{full_paragraph}\n")

        break