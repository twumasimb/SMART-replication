# Import libraries
import os
import torch
from trl import SFTTrainer, SFTConfig
from time import perf_counter
from transformers import GenerationConfig
from datasets import load_dataset, Dataset, load_from_disk
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments

def formatted_prompt(question) -> str:
    return f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def formatted_train(input, response):
    return f"<|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>"

# Step 3: Define a mapping function to apply the formatting
def format_data(example):
    example["text"] = formatted_train(example["prompt"], example["response"])
    return example

# get the tokenizer and model 
def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token=tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, 
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer

def generate_response(user_input, model, tokenizer):
    prompt = formatted_prompt(user_input)
    # inputs = tokenizer([prompt], return_tensors="pt")
    generation_config = GenerationConfig(penalty_alpha=0.6, do_sample=True,
        top_k=5, temperature=0.5, repetition_penalty=1.2, 
        max_new_tokens=60, pad_token_id=tokenizer.eos_token_id
    )
    start_time = perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = (tokenizer.decode(outputs[0], skip_special_tokens=True))
    generation_time = perf_counter() - start_time
    print(f"Time taken for inference: {round(generation_time, 2)} seconds.")

    return response

def main():
    # Define the model you want to finetune
    model_name = 'Llama-3.2-3B'
    model_id = f"meta-llama/{model_name}"

    run_name = "Initial Run"

    # Name of the new model
    output_model=model_name
    output_dir=f"models/{model_name}"

    # Load dataset from memory
    ds = load_from_disk('/mnt/DATA/datasets/final_dataset')

    # Define the model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_id=model_id)

    # Get the training data
    train_dataset = ds['train'].map(format_data)
    num_epochs = 3
    batch_size = 2

    # Calculate the number of iterations
    num_samples = len(train_dataset)
    num_iterations = num_epochs * (num_samples // batch_size)
    

    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )

    training_arguments = TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        optim="adamw_8bit",
        run_name=run_name,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=1,
        num_train_epochs=num_epochs,
        max_steps= num_iterations,
        fp16=True,
        push_to_hub=False,
        seed=3407
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024  
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()