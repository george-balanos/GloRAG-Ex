import random
import pandas as pd
import re
import json
from typing import List


class PoolNoiseSelector:
    def __init__(self):
        self.paragraph_pool = []
        self.sentence_pool = []

    def build_csv_pools_from_kg(self, kg_json_path: str, out_prefix: str = "kg_noise"):
        with open(kg_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_paragraphs = []

        def extract(x):
            if isinstance(x, str):
                raw_paragraphs.append(x)

            elif isinstance(x, dict):
                text_keys = {"text", "chunk", "content", "value"}
                for k, v in x.items():
                    if k in text_keys and isinstance(v, str):
                        raw_paragraphs.append(v)
                    else:
                        extract(v)

            elif isinstance(x, list):
                for item in x:
                    extract(item)

        extract(data)
        clean_paragraphs = []
        clean_sentences = []

        for blob in raw_paragraphs:
            for p in self._split_blob_into_paragraphs(blob) or [blob]:
                if not self._clean(p):
                    continue
                clean_paragraphs.append(p)

                for s in self._split_blob_into_sentences(p):
                    if self._clean(s):
                        clean_sentences.append(s)

        pd.DataFrame({"paragraph": clean_paragraphs}).to_csv(
            f"{out_prefix}_paragraphs.csv", index=False
        )
        pd.DataFrame({"sentence": clean_sentences}).to_csv(
            f"{out_prefix}_sentences.csv", index=False
        )

        print(f"[KG LOADER] extracted {len(raw_paragraphs)} raw strings "
              f"from {kg_json_path}")
        print(f"[OK] Saved {out_prefix}_paragraphs.csv "
              f"({len(clean_paragraphs)} paragraphs)")
        print(f"[OK] Saved {out_prefix}_sentences.csv "
              f"({len(clean_sentences)} sentences)")

    def _clean(self, x: str) -> bool:
        if not isinstance(x, str):
            return False
        x = x.strip()
        if len(x) < 30:
            return False
        if len(x) > 1000:
            return False
        if "default:extract" in x:
            return False
        if "doc-" in x or "chunk-" in x:
            return False
        if "Question:" in x or "Answers:" in x or "Context:" in x:
            return False
        if re.fullmatch(r"[a-f0-9\-]{20,}", x):
            return False
        if len([c for c in x if c.isalpha()]) < 10:
            return False
        return True

    def _split_blob_into_paragraphs(self, blob: str) -> List[str]:

        if not isinstance(blob, str):
            return []
        return [p.strip() for p in blob.split("\n\n") if p.strip()]

    def _split_blob_into_sentences(self, blob: str) -> List[str]:

        if not isinstance(blob, str):
            return []

        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", blob.strip())
        return [p.strip() for p in parts if p.strip()]

    def load_csv_pools(self, para_csv: str, sent_csv: str):
        df_p = pd.read_csv(para_csv)
        df_s = pd.read_csv(sent_csv)

        raw_paragraph_blobs = df_p["paragraph"].dropna().tolist()
        split_paragraphs = []
        for blob in raw_paragraph_blobs:
            split_paragraphs.extend(self._split_blob_into_paragraphs(blob))
        self.paragraph_pool = [x for x in split_paragraphs if self._clean(x)]

        raw_sentence_blobs = df_s["sentence"].dropna().tolist()
        split_sentences = []
        for blob in raw_sentence_blobs:
            split_sentences.extend(self._split_blob_into_sentences(blob))
        self.sentence_pool = [x for x in split_sentences if self._clean(x)]

        print(f"[POOL CLEAN] paragraphs loaded: {len(self.paragraph_pool)} "
              f"(from {len(raw_paragraph_blobs)} raw CSV rows)")
        print(f"[POOL CLEAN] sentences loaded: {len(self.sentence_pool)} "
              f"(from {len(raw_sentence_blobs)} raw CSV rows)")

    def get_units(self, text: str, mode: str) -> List[str]:
        if mode == "sentence":
            return [s.strip() for s in text.split(".") if s.strip()]
        elif mode == "paragraph":
            return [p.strip() for p in text.split("\n\n") if p.strip()]
        else:
            raise ValueError("mode must be sentence or paragraph")

    def inject_noise(self, text: str, noise_percent: float = 0.2, mode: str = "sentence") -> str:
        if not self.paragraph_pool or not self.sentence_pool:
            raise ValueError("Load CSV pools first")

        units = self.get_units(text, mode)
        n = len(units)
        if n == 0:
            return text

        pool = self.sentence_pool if mode == "sentence" else self.paragraph_pool


        k = max(1, int(round(n * noise_percent)))
        k = min(k, len(pool))

        noise_items = random.sample(pool, k)

        min_words = 5 if mode == "sentence" else 8
        noise_items = [
            s for s in noise_items
            if isinstance(s, str) and len(s.split()) >= min_words
        ]


        combined = units.copy()
        insert_positions = sorted(
            random.sample(range(len(combined) + 1), len(noise_items))
        )
        for pos, noise in zip(reversed(insert_positions), reversed(noise_items)):
            combined.insert(pos, noise)

        if mode == "sentence":
            cleaned = [u.rstrip(".") for u in combined]
            return ". ".join(cleaned) + "."
        else:
            return "\n\n".join(combined)


if __name__ == "__main__":
    selector = PoolNoiseSelector()
    selector.build_csv_pools_from_kg(
        "../../../xylotian_storage/kv_store_text_chunks.json",
        out_prefix="kg_noise"
    )

    selector.load_csv_pools(
        "kg_noise_paragraphs.csv",
        "kg_noise_sentences.csv"
    )

    text = """Graph neural networks are powerful models. They can encode structural relationships.
            Retrieval augmented generation improves factuality. It reduces hallucination in LLMs.
            Knowledge graphs provide structured context for downstream reasoning tasks."""

    print("\n================ SENTENCE MODE ================\n")
    print(selector.inject_noise(
        text,
        noise_percent=0.5,
        mode="sentence"
    ))

    print("\n================ PARAGRAPH MODE ================\n")
    print(selector.inject_noise(
        text,
        noise_percent=0.1,
        mode="paragraph"
    ))