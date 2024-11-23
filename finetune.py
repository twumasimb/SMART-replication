"""
Fine-tuning script for a causal language model using PEFT (Parameter-Efficient Fine-Tuning).

This script supports:
- Data preprocessing and formatting
- Quantized model loading
- Fine-tuning using LoRA
- Configurable training parameters
- Inference and evaluation

Author: Twumasi
"""

import os
import torch
from time import perf_counter
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig
)
from trl import SFTTrainer
from peft import LoraConfig


def formatted_prompt(question: str) -> str:
    """
    Format a question for the model as input.
    Args:
        question (str): The user's question or prompt.

    Returns:
        str: Formatted prompt string.
    """
    return f"<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"


def formatted_train(input_text: str, response_text: str) -> str:
    """
    Format input-output pairs for training.
    Args:
        input_text (str): User's input text.
        response_text (str): Assistant's response text.

    Returns:
        str: Formatted training example.
    """
    return f"<|start_header_id|>user<|end_header_id|>\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{response_text}<|eot_id|>"


def format_data(example):
    """
    Dataset formatting function to prepare examples for training.
    Args:
        example (dict): Dataset example containing 'prompt' and 'response'.

    Returns:
        dict: Formatted example with 'text' field.
    """
    example["text"] = formatted_train(example["prompt"], example["response"])
    return example


def get_model_and_tokenizer(model_id: str):
    """
    Load the pre-trained model and tokenizer with quantization configuration.
    Args:
        model_id (str): Model ID from Hugging Face or local directory.

    Returns:
        tuple: Tokenizer and model objects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        device_map='auto'
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model, tokenizer


def generate_response(user_input: str, model, tokenizer):
    """
    Generate a response from the model for a given user input.
    Args:
        user_input (str): Input query from the user.
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.

    Returns:
        str: Model's response.
    """
    prompt = formatted_prompt(user_input)
    generation_config = GenerationConfig(
        penalty_alpha=0.6,
        do_sample=True,
        top_k=5,
        temperature=0.5,
        repetition_penalty=1.2,
        max_new_tokens=60,
        pad_token_id=tokenizer.eos_token_id
    )
    start_time = perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generation_time = perf_counter() - start_time
    print(f"Time taken for inference: {round(generation_time, 2)} seconds.")
    return response


def load_datasets(dataset_path: str):
    """
    Load and preprocess the datasets.
    Args:
        dataset_path (str): Path to the preprocessed dataset.

    Returns:
        tuple: Train and validation datasets.
    """
    ds = load_from_disk(dataset_path)
    train_dataset = ds['train'].map(format_data, batched=True)
    val_dataset = ds['validation'].shuffle(seed=42).select(range(5000)).map(format_data, batched=True)
    return train_dataset, val_dataset


def get_training_arguments(output_model: str, run_name: str) -> TrainingArguments:
    """
    Define training arguments for the SFTTrainer.
    Args:
        output_model (str): Directory to save the fine-tuned model.
        run_name (str): Name for the training run.

    Returns:
        TrainingArguments: Configuration for training.
    """
    return TrainingArguments(
        output_dir=output_model,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        optim="adamw_8bit",
        run_name=run_name,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        num_train_epochs=3,
        fp16=True,
        push_to_hub=False,
        seed=3407
    )


def train_model(model, tokenizer, train_dataset, val_dataset, training_arguments, peft_config):
    """
    Train the model using SFTTrainer.
    Args:
        model: Pre-trained model.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        training_arguments: Training configuration.
        peft_config: PEFT configuration for LoRA.

    Returns:
        SFTTrainer: The trainer instance after training.
    """
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_arguments,
        tokenizer=tokenizer,
        packing=False,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024
    )
    trainer.train()
    return trainer


def main():
    """
    Main function to execute the fine-tuning process.
    """
    # Model configuration
    model_name = 'Llama-3.2-1B'
    model_id = f"meta-llama/{model_name}"
    output_model = f"models/{model_name}"
    dataset_path = '/mnt/DATA/datasets/final_dataset'
    run_name = "Initial Run"

    # Load datasets
    train_dataset, val_dataset = load_datasets(dataset_path)

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_id=model_id)

    # Configure PEFT
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get training arguments
    training_arguments = get_training_arguments(output_model, run_name)

    # Train model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, training_arguments, peft_config)

    # Save the model and tokenizer
    model.save_pretrained(output_model)
    tokenizer.save_pretrained(output_model)


if __name__ == '__main__':
    main()
