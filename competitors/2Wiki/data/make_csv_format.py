import pandas as pd
import re

INPUT_FILE = "./2wiki_dev.csv"
OUTPUT_FILE = "./2wiki_fixed.csv"

def longest_supporting_paragraph(text):
    if pd.isna(text):
        return ""

    paragraphs = [p.strip() for p in str(text).split("|||")]
    paragraphs = [p for p in paragraphs if p]

    if not paragraphs:
        return ""

    return max(paragraphs, key=len)

# Read pipe-separated file
df = pd.read_csv(
    INPUT_FILE,
    sep="|",
    quotechar='"',
    engine="python"
)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())

out_df = pd.DataFrame({
    "id": range(1, len(df) + 1),
    "question": df["Question"],
    "answer": df["Supporting Paragraph"].apply(longest_supporting_paragraph)
})

out_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(out_df)} rows to {OUTPUT_FILE}")