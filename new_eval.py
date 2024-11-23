import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

# Load the MMLU dataset (ensure you have downloaded it beforehand)
# Example: {'question': "What's 2 + 2?", 'options': ['3', '4', '5', '6'], 'answer': '4'}
def load_mmlu_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load the local LLM model and tokenizer
def load_local_llm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# Generate a response for a given prompt
def generate_response(prompt, tokenizer, model, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=0.7,  # Adjust for more/less randomness
            num_return_sequences=1,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Evaluate exact match accuracy
def evaluate_mmlu(dataset, tokenizer, model):
    total = len(dataset)
    correct = 0

    for i, entry in enumerate(dataset):
        question = entry['question']
        options = "\n".join([f"({chr(65 + idx)}) {opt}" for idx, opt in enumerate(entry['choices'])])
        prompt = f"Question: {question}\nOptions:\n{options}\nAnswer:"

        response = generate_response(prompt, tokenizer, model)

        # Extract the predicted answer (assume it's the first letter in response)
        predicted_answer = response.strip().split()[0]  # Take the first token (e.g., 'A', 'B', 'C', 'D')
        correct_answer = entry['answer']

        if predicted_answer == correct_answer:
            correct += 1

        # Logging for debugging
        print(f"[{i + 1}/{total}] Question: {question}")
        print(f"Predicted: {predicted_answer}, Correct: {correct_answer}\n")

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    # Path to the MMLU dataset
    mmlu_file = "path/to/mmlu_dataset.json"

    # Path to the local model (replace with your local LLM model name)
    model_name = "path/to/local-llm"

    dataset = load_dataset("cais/mmlu", "abstract_algebra")
    dataset = dataset['test']

    # Load dataset and model
    # dataset = load_mmlu_dataset(mmlu_file)
    tokenizer, model = load_local_llm(model_name)

    # Evaluate the model
    accuracy = evaluate_mmlu(dataset, tokenizer, model)

    print(f"Exact Match Accuracy: {accuracy:.2%}")
