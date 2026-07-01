import random
import pandas as pd
import re
import json
from typing import List, Tuple


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
        clean_sentences  = []

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

    def load_csv_pools(
        self,
        para_csv: str | None,
        sent_csv: str | None,
    ):
        if sent_csv:
            df_s = pd.read_csv(sent_csv)
            raw_sentence_blobs = df_s["sentence"].dropna().tolist()
            split_sentences = []
            for blob in raw_sentence_blobs:
                split_sentences.extend(self._split_blob_into_sentences(blob))
            self.sentence_pool = [x for x in split_sentences if self._clean(x)]
            print(f"[POOL CLEAN] sentences loaded: {len(self.sentence_pool)} "
                  f"(from {len(raw_sentence_blobs)} raw CSV rows)")

        if para_csv:
            df_p = pd.read_csv(para_csv)
            raw_paragraph_blobs = df_p["paragraph"].dropna().tolist()
            split_paragraphs = []
            for blob in raw_paragraph_blobs:
                split_paragraphs.extend(self._split_blob_into_paragraphs(blob))
            self.paragraph_pool = [x for x in split_paragraphs if self._clean(x)]
            print(f"[POOL CLEAN] paragraphs loaded: {len(self.paragraph_pool)} "
                  f"(from {len(raw_paragraph_blobs)} raw CSV rows)")

    def get_units(self, text: str, mode: str) -> List[str]:
        if mode == "sentence":
            return [s.strip() for s in text.split(".") if s.strip()]
        elif mode == "paragraph":
            return [p.strip() for p in text.split("\n\n") if p.strip()]
        else:
            raise ValueError("mode must be sentence or paragraph")

    def inject_noise(
        self,
        text: str,
        noise_percent: float = 0.2,
        mode: str = "sentence",
        seed: int | None = None,
    ) -> Tuple[str, set]:
        """Inject noise units at random positions.

        Returns
        -------
        noisy_context   : str  – the context with noise inserted
        noise_positions : set  – integer indices (into the final list)
                                 that correspond to injected noise units
        """
        # only check the pool we actually need
        if mode == "sentence" and not self.sentence_pool:
            raise ValueError("Sentence pool empty — pass --pool-sent.")
        if mode == "paragraph" and not self.paragraph_pool:
            raise ValueError("Paragraph pool empty — pass --pool-para.")

        rng   = random.Random(seed)
        units = self.get_units(text, mode)
        n     = len(units)
        if n == 0:
            return text, set()

        pool = self.sentence_pool if mode == "sentence" else self.paragraph_pool

        # exclude anything already present in the original context
        orig_texts    = set(units)
        filtered_pool = [item for item in pool if item not in orig_texts]

        if not filtered_pool:
            print("[inject_noise] pool exhausted after filtering originals; "
                  "returning unchanged context.")
            return text, set()

        k = max(1, int(round(n * noise_percent)))
        k = min(k, len(filtered_pool))

        # pool items already passed _clean() on load, no further filtering needed
        noise_items = rng.sample(filtered_pool, k)

        combined = units.copy()

        insert_positions = sorted(
            rng.sample(
                range(len(combined) + 1),
                min(len(noise_items), len(combined) + 1),
            )
        )

        # Insert right-to-left so earlier insertions don't shift later positions
        for pos, noise in zip(reversed(insert_positions), reversed(noise_items)):
            combined.insert(pos, noise)

        # Replay insertions left-to-right on a boolean tracker to get
        # final positions — exact even if noise text duplicates original
        tracker: list[bool] = [False] * len(units)
        for pos in insert_positions:
            tracker.insert(pos, True)   # True = noise

        final_noise_positions = {i for i, is_noise in enumerate(tracker) if is_noise}

        if mode == "sentence":
            cleaned  = [u.rstrip(".") for u in combined]
            rendered = ". ".join(cleaned) + "."
        else:
            rendered = "\n\n".join(combined)

        return rendered, final_noise_positions


if __name__ == "__main__":
    selector = PoolNoiseSelector()
    selector.build_csv_pools_from_kg(
        "/home/gbalanos/GloRAG-Ex/code/KGs/lightrag/2wiki/kv_store_text_chunks.json",
        out_prefix="/home/gbalanos/GloRAG-Ex/competitors/RAGEX-RAGE-SHAPLEY/noise_pool/2wiki_kg_noise"
    )