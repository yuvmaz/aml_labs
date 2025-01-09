from accelerate import Accelerator

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    AutoPeftModelForCausalLM,
    prepare_model_for_kbit_training,
    get_peft_model,
)

import os, torch, mlflow
from datasets import load_dataset, Dataset, ReadInstruction
from trl import SFTTrainer, setup_chat_format
import argparse
import os
import mlflow
import torch

def prepare_data(finetune_dataset, num_data_rows, hf_cache, base_model, eval_size):
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=hf_cache)

    dataset = load_dataset(finetune_dataset, split="all", cache_dir=hf_cache)
    dataset = dataset.shuffle(seed=65).select(range(num_data_rows))

    def format_chat_template(row):
        row_json = [{"role": "user", "content": row["Patient"]},
                   {"role": "assistant", "content": row["Doctor"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row


    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ ' ' }}{% endif %}{{ message['content'] }}{% if not loop.last %}{{ '  ' }}{% endif %}{% endfor %}{{ eos_token }}"

    dataset = dataset.map(
        format_chat_template,
        num_proc=4,
    )

    return dataset.train_test_split(test_size=eval_size)

def do_training(base_model, eval_size, finetune_dataset, finetuned_model, cache_dir, num_epochs, batch_size):

    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    device_index = Accelerator().process_index
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=qlora_config,
        device_map = {"": device_index},
        attn_implementation="eager",
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model, tokenizer = setup_chat_format(model, tokenizer)


    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM", 
    )
    model = get_peft_model(model, peft_config)




    training_arguments = TrainingArguments(
        output_dir=finetuned_model,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=num_epochs,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=50,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        group_by_length=True,
        report_to='azure_ml'
    )


    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        max_seq_length=512,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing= False,
    )

    OUTPUTS = "./outputs/"
    trainer.train()
    trainer.save_model(OUTPUTS + finetuned_model)  
    peft_config.save_pretrained(OUTPUTS + finetuned_model + "_config")
    model.merge_and_unload().save_pretrained(OUTPUTS + finetuned_model + "_full")

    return model, tokenizer


def get_answer(model, tokenizer):
    messages = [
       {
            "role": "user",
            "content": "Hello doctor, I get red blotches on my skin whenever I'm next to a cat." + 
            " What can I do?"
       }
    ]  

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to('cuda:0')

    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return text


device_id = torch.cuda.current_device()

if __name__ == '__main__':

    import sys
    print(f"Arguments: {sys.argv}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="NousResearch/Meta-Llama-3-8B", help="Model to fine-tune")
    parser.add_argument("--dataset", type=str, default="ruslanmv/ai-medical-chatbot", help="Dataset to use for finetuning")
    parser.add_argument("--num_data_rows", type=int, default=1000, help="Number of data rows to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (both train and eval)")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--finetuned_model", type=str, default="llama3-8b-chat-doctor")
    parser.add_argument("--eval_size", type=float, default=0.1, help="Fraction of evaluation set")
    parser.add_argument("--hf_cache", type=str, default="/mounts/models", help="Location of huggingface cache")

    args = parser.parse_args()

    print("Starting run...")
    print("Script path: " + os.path.dirname(os.path.realpath(__file__)))

    dataset = prepare_data(args.dataset, args.num_data_rows, args.hf_cache, args.base_model, args.eval_size)
    model, tokenizer = do_training(args.base_model, args.eval_size, dataset, \
        args.finetuned_model, args.hf_cache, args.num_epochs, args.batch_size)


    if Accelerator().process_index == 0:
        print(get_answer(model, tokenizer))










