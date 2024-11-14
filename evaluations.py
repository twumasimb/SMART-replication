import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

dataset = load_dataset("cais/mmlu", "abstract_algebra")
dataset = dataset['test']

# model_name = "models/llama-2-7b-bnb-4bit-finetuned"  # Replace with your model's name or path
# model_name = 'RAL-llama3.2-1B-Final Run'
# model_name = 'unsloth/llama-2-7b-bnb-4bit'
# model_name = 'meta-llama/Llama-3.2-1B'
# model_name = 'models/Llama-3.2-1B'
model_name = 'models/mistral-7b-bnb-4bit-finetuned-v2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def evaluate_mmlu(model, tokenizer, dataset):
    correct = 0
    total = len(dataset)
    
    pbar = tqdm(range(len(dataset)))
    for item in dataset:
        # Prepare the input
        question = item["question"]
        choices = item['choices']
        correct_answer = item["answer"]
        
        # Tokenize input
        inputs = [question + " " + choice for choice in choices]
        encodings = tokenizer(inputs, padding=True, return_tensors="pt").to(model.device)
        
        # Get model predictions
        outputs = model(**encodings)
        logits = outputs.logits
        
        # Calculate the scores for each choice
        choice_scores = logits[:, -1, :].mean(dim=-1)
        predicted_answer = torch.argmax(choice_scores).item()
        
        # Compare with correct answer
        if predicted_answer == correct_answer:
            correct += 1
        print(f"Predicted Answer: {predicted_answer} | Correct Answer: {correct_answer}")

        pbar.update(1)
    accuracy = correct / total
    return accuracy

if __name__== "__main__":
    accuracy = evaluate_mmlu(model, tokenizer, dataset)
    print(f"MMLU Accuracy: {accuracy * 100:.2f}%")

