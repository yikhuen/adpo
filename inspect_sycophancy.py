import json
from huggingface_hub import hf_hub_download

try:
    local_path = hf_hub_download(
        repo_id="meg-tong/sycophancy-eval",
        filename="are_you_sure.jsonl",
        repo_type="dataset"
    )
    
    with open(local_path, "r", encoding="utf-8") as f:
        # Read first line only to inspect structure
        first_line = f.readline()
        row = json.loads(first_line)
        print("Keys available:", list(row.keys()))
        print("\nFull example content:")
        print(json.dumps(row, indent=2))

except Exception as e:
    print(f"Error: {e}")

