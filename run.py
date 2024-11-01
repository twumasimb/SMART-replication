from transformers import Trainer, TrainingArguments
from peft import PeftModel, PeftConfig

# ...existing code...

def peft_finetuning(model, train_dataset, eval_dataset, output_dir, peft_config):
    """
    Perform PEFT finetuning on the given model.

    Args:
        model: The pre-trained model to be fine-tuned.
        train_dataset: The dataset for training.
        eval_dataset: The dataset for evaluation.
        output_dir: The directory where the model checkpoints will be saved.
        peft_config: The configuration for PEFT.
    """
    # Wrap the model with PEFT
    peft_model = PeftModel(model, peft_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model(output_dir)

# ...existing code...


def preprocess_function(examples):
    prompts_responses=[p+" "+r for p, r in zip(examples["prompt"], examples["response"])]
    prompts_responses_tokenized=tokenizer(prompts_responses, truncation=True, max_length=max_seq_length)
    prompts_tokenized=tokenizer(examples["prompt"], truncation=True, max_length=max_seq_length)
    all_labels=copy.deepcopy(prompts_responses_tokenized["input_ids"])
    prompts_len=[len(prompt) for prompt in prompts_tokenized["input_ids"]]
    for labels, prompt_len in zip(all_labels, prompts_len):
        labels[:prompt_len]=[IGNORE_INDEX]*prompt_len
    result={k: v for k, v in prompts_responses_tokenized.items()}
    result["labels"]=all_labels
    return result

preprocessed_dataset=raw_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=preprocessing_num_workers,
    load_from_cache_file=not overwrite_cache,
    remove_columns=raw_dataset_column_names,
    desc="Preprocessing the raw dataset",
)

train_dataset=preprocessed_dataset["train"]
eval_dataset=preprocessed_dataset["validation"]

# DataLoaders creation
data_collator=DataCollatorForInstructionTuning(tokenizer)
train_dataloader=DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, pin_memory=True, num_workers=8
)
eval_dataloader=DataLoader(
    eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size, pin_memory=True, num_workers=8
)