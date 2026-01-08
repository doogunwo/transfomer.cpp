# src/front.py
import argparse
from pathlib import Path

from runtime import Runtime, RuntimeConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/llama-2-7b-chat.Q4_K_M.gguf", help="path to gguf model")
    ap.add_argument("--prompt", required=True, help="input prompt")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--show_kv_suggestion", action="store_true",
                    help="print KV-cache config suggestion after loading model")
    args = ap.parse_args()

    rt = Runtime(RuntimeConfig(verbose=args.verbose))
    info = rt.load_model(args.model)

    print("[Model Summary]")
    for k, v in info.items():
        print(f"  {k}: {v}")

    if args.show_kv_suggestion:
        sugg = rt.kv_cache_suggestion()
        print("\n[KV Cache Suggestion]")
        for k, v in sugg.items():
            print(f"  {k}: {v}")

    out = rt.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print("\n" + out)


if __name__ == "__main__":
    main()
