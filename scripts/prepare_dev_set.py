import os
import json
import typer
from datasets import load_dataset

app = typer.Typer()


@app.command()
def main(dataset: str = typer.Option(...), size: int = typer.Option(200), out: str = typer.Option("data/dev.jsonl")):
    ds = load_dataset(dataset, split="test_prefs")
    n = min(size, len(ds))
    ds = ds.select(range(n))
    prompts = []
    for row in ds:
        messages = row.get("chosen") or row.get("messages") or []
        user_msg = [m for m in messages if m.get("role") == "user"]
        if not user_msg:
            continue
        prompts.append({"prompt": user_msg[0].get("content", "")})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    app()
