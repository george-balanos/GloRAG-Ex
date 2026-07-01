# import pandas as pd

# df = pd.read_csv("2wiki_fixed.csv")
# df = df.drop(columns=["context"])
# df.to_csv("qa_data_2wiki.csv", index=False)

# print(f"Saved {len(df)} rows to qa_2wiki.csv")

import pandas as pd

def longest_answer(text):
    if pd.isna(text):
        return ""
    options = [a.strip() for a in str(text).split("|||") if a.strip()]
    if not options:
        return ""
    return max(options, key=len)

df = pd.read_csv("qa_data_2wiki.csv")
df["answer"] = df["answer"].apply(longest_answer)
df.to_csv("qa_data_2wiki.csv", index=False)

print(f"Saved {len(df)} rows to qa_2wiki.csv")