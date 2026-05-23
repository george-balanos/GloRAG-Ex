import json
import sys

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_ft_to_ff(file1_path, file2_path):
    data1 = load_json(file1_path)
    data2 = load_json(file2_path)

    results1 = data1.get("results", {})
    results2 = data2.get("results", {})

    degraded = {}
    for qid, entry1 in results1.items():
        if entry1.get("case") == "ft":
            entry2 = results2.get(qid)
            if entry2 and entry2.get("case") == "ff":
                degraded[qid] = entry2

    return degraded


def build_output(degraded_results):
    case_ids = list(degraded_results.keys())

    output = {
        "summary": {
            "total": len(case_ids),
            "tt": 0,
            "tf": 0,
            "ft": 0,
            "ff": len(case_ids),
            "llm_accuracy": 0.0,
            "rag_accuracy": 0.0,
        },
        "cases": {
            "tt": [],
            "tf": [],
            "ft": [],
            "ff": case_ids,
        },
        "results": degraded_results,
    }

    return output


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python find_ft_to_ff.py <file1.json>(FT) <file2.json>(FF) [output.json]")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) == 4 else "ft_to_ff.json"

    degraded = find_ft_to_ff(file1_path, file2_path)

    if not degraded:
        print("No questions changed from 'ft' to 'ff' between the two files.")
        sys.exit(0)

    output = build_output(degraded)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(degraded)} degraded question(s) to '{output_path}'.")


if __name__ == "__main__":
    main()