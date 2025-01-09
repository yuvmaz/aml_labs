from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

import argparse

if __name__ == '__main__':

    import sys
    print(f"Arguments: {sys.argv}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="NousResearch/Meta-Llama-3-8B", help="Model to fine-tune")
    parser.add_argument("--hf_cache", type=str, default="/mounts/models", help="Location of huggingface cache")
      
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.hf_cache)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.hf_cache
    )