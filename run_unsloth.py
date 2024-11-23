import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, load_from_disk
from unsloth import FastLanguageModel

def formatted_prompt(question) -> str:
    return f"prompt:\n{question}"

def formatted_train(input, response):
    return f"prompt:\n{input}\n\nresponse:\n{response}<|eot_id|>"

# Step 3: Define a mapping function to apply the formatting
def format_data(example):
    example["text"] = formatted_train(example["prompt"], example["response"])
    return example

dir = '/mnt/DATA/datasets'
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
dataset = load_from_disk(f"{dir}/final_dataset") # Get dataset
dataset = dataset['train'].map(format_data)
# model_  = 'llama-2-7b-bnb-4bit'
model_ = 'mistral-7b-bnb-4bit'

# Load Llama model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"unsloth/{model_}", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
      per_device_train_batch_size = 2,
      gradient_accumulation_steps = 4,
      warmup_steps = 10,
      max_steps = 60,
      fp16 = not torch.cuda.is_bf16_supported(),
      bf16 = torch.cuda.is_bf16_supported(),
      logging_steps = 1,
      output_dir = "outputs-version-2",
      optim = "adamw_8bit",
      seed = 3407,
  ),
)
trainer.train()

output_dir = f"models/{model_}-finetuned-v2"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)