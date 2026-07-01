from datasets import load_dataset

ds = load_dataset("dgslibisey/MuSiQue")

ds["train"].to_json("musique_final.json")