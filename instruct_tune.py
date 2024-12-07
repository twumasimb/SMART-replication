import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, set_seed, TrainingArguments, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading a base model (llama-3-1B)
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Create a function to generate text from the given model

def generate(prompt:str, max_new_tokens:int=100):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated_tokens = model.generate(
        input_ids,
        max_new_tokens = max_new_tokens,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id =  tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Define Llama-style chat template with special tokens
llama_chat_template = """<|begin_of_text|>{% for message in messages %}
{% if message['role'] == 'system' %}
<|start_header_id|>system<|end_header_id|>{{ message['content'] }}<|eom_id|>
{% elif message['role'] == 'user' %}
<|start_header_id|>user<|end_header_id|>{{ message['content'] }}<|eom_id|>
{% elif message['role'] == 'assistant' %}
<|start_header_id|>assistant<|end_header_id|>{{ message['content'] }}<|eom_id|>
{% endif %}{% endfor %}<|eot_id|>"""

def format_data_for_template(data):
    """
    Convert a prompt-response pair into a dictionary containing messages list.
    
    Args:
        data (dict): Dictionary containing 'prompt' and 'response' keys
        
    Returns:
        dict: Dictionary with 'messages' key containing the formatted messages
    """
    messages = [
        {
            "role": "user",
            "content": data['prompt']
        },
        {
            "role": "assistant",
            "content": data['response']
        }
    ]
    
    return {"messages": messages}

# Example usage

tokenizer.chat_template = llama_chat_template

# Setting padding token to eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id

# Setting padding token to eos_token
tokenizer.pad_token_id =  tokenizer.eos_token_id

def format_data(example):
    
    # Convert the data to meet the chat template structure
    chat = [{"role": message['role'], "content": message['content']} for message in example['messages']]
    
    # apply the chat template from the tokenizer
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

    #Tokenize using the standard approach.
    tokenized_output = tokenizer(formatted_chat, add_special_tokens=False, padding="max_length", max_length=512, truncation=True)

    return tokenized_output

# Creating a dataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #mlm --> Masked Language Modeling

def main():
    # Map the dataset
    ds = load_from_disk("/mnt/DATA/datasets/final_dataset")
    formatted_dataset = ds.map(format_data_for_template, num_proc=16).remove_columns(['prompt', 'response'])
    tokenized_formatted_data = formatted_dataset.map(format_data, num_proc=16).remove_columns('messages')

    # splitting the dataset into training and evaluation
    set_seed(123)
    train_dataset = tokenized_formatted_data['train']
    test_set = tokenized_formatted_data['validation'].train_test_split(train_size=10000, test_size=1000)

    # Setting up for training
    LOG_DIR = "output/logging"
    OUTPUT_DIR = "output/model"

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, 
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        auto_find_batch_size=True,
        gradient_accumulation_steps=8,
        warmup_steps=1,
        weight_decay=0.01,
        logging_dir=LOG_DIR,
        logging_steps=1,
        eval_strategy="epoch",
        lr_scheduler_type="linear",
        bf16=True, 
        gradient_checkpointing=True,
        save_steps=1000,
        learning_rate=8.5e-6,
        # use_cache=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_set['test'],
        data_collator=data_collator
    )
    # trainer.evaluate()
    trainer.train()

if __name__ == "__main__":
    main()