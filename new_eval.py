import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch

# Load the local LLM model and tokenizer
def load_local_llm(model_name):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model #tokenizer, model

# Generate a response for a given prompt
def generate_response(prompt, tokenizer, model, max_length=150):
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

def formatted_prompt(question) -> str:
    return f"<|start_header_id|>user<|end_header_id|>\n{question}. Think step by step: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# Evaluate exact match accuracy
def evaluate_mmlu(dataset, tokenizer, model):
    total = len(dataset)
    correct = 0

    for i, entry in enumerate(dataset):
        question = entry['question']
        options = "\n".join([f"({chr(65 + idx)}) {opt}" for idx, opt in enumerate(entry['choices'])])
        prompt = f"Question: {question}\nOptions:\n{options}\nAnswer:"
        prompt = formatted_prompt(prompt)

        response = generate_response(prompt, tokenizer, model)
        print("Response_________________")
        print(response)
        print("_________________________")

        # Extract the predicted answer (assume it's the first letter in response)
        predicted_answer = response.strip().split()[0]  # Take the first token (e.g., 'A', 'B', 'C', 'D')
        correct_answer = entry['answer']
        print(f"Correct Answer: {correct_answer}")

        if predicted_answer == correct_answer:
            correct += 1

        # Logging for debugging
        # print(f"[{i + 1}/{total}] Question: {question}\n")
        # print(f"Predicted: {predicted_answer}, Correct: {correct_answer}\n")

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    
    # Path to the local model (replace with your local LLM model name)
    # model_name = f"models/Llama-3.2-3B"
    model_name = f"output/model/checkpoint-1000"

    dataset = load_dataset("cais/mmlu", "abstract_algebra")
    dataset = dataset['dev']

    # Load dataset and model
    # dataset = load_mmlu_dataset(mmlu_file)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer, model = load_local_llm(model_name)

    # Evaluate the model
    accuracy = evaluate_mmlu(dataset, tokenizer, model)

    print(f"Exact Match Accuracy: {accuracy:.2%}")
