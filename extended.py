import os
import math
import collections
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

def calculate_perplexity_long(text, model, tokenizer, max_seq_length=1024):
    tokens = tokenizer.tokenize(text.lower())
    num_chunks = math.ceil(len(tokens) / max_seq_length)
    perplexities = []
    for i in range(num_chunks):
        start_idx = i * max_seq_length
        end_idx = (i + 1) * max_seq_length
        chunk_tokens = tokens[start_idx:end_idx]
        input_ids = tokenizer.encode(chunk_tokens, return_tensors="pt")
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids).loss
        perplexities.append(math.exp(loss))
    return math.exp(sum(perplexities) / len(perplexities))

def calculate_burstiness(tokens):
    token_counts = collections.Counter(tokens)
    num_tokens = len(tokens)
    return (num_tokens - len(token_counts)) / num_tokens

def analyze_text(sample, model, tokenizer):
    ai_generated_text = sample["generated_intro"]
    human_text = sample["wiki_intro"]
    ai_tokens = tokenizer.tokenize(ai_generated_text.lower())
    human_tokens = tokenizer.tokenize(human_text.lower())
    ai_perplexity = calculate_perplexity_long(ai_generated_text, model, tokenizer)
    human_perplexity = calculate_perplexity_long(human_text, model, tokenizer)
    ai_burstiness = calculate_burstiness(ai_tokens)
    human_burstiness = calculate_burstiness(human_tokens)
    return ai_perplexity, ai_burstiness, human_perplexity, human_burstiness, ai_generated_text, human_text

def main():
    print("Sample testing live performance:")
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    dataset = load_dataset("aadityaubhat/GPT-wiki-intro")
    total_samples = 0
    total_ai_correct = 0
    total_human_correct = 0
    ai_accuracy_data = []
    human_accuracy_data = []
    cumulative_accuracy_data = []

    with open("results_aitexts.txt", "w") as results_ai_file, open("results_humantexts.txt", "w") as results_human_file:
        with ThreadPoolExecutor() as executor:
            future_to_sample = {}
            for sample in dataset["train"]:
                if total_samples >= 6700:
                    break
                future = executor.submit(analyze_text, sample, model, tokenizer)
                future_to_sample[future] = sample
                total_samples += 1

            for future in tqdm(concurrent.futures.as_completed(future_to_sample), total=len(future_to_sample), desc="Analyzing"):
                sample = future_to_sample[future]
                try:
                    ai_perplexity, ai_burstiness, human_perplexity, human_burstiness, ai_generated_text, human_text = future.result()
                except Exception as exc:
                    print(f'An exception occurred: {exc}')
                    continue

                perplexity_threshold = 15.0
                burstiness_threshold = 0.45
                verdict_ai = "likely AI-generated" if ai_perplexity > perplexity_threshold and ai_burstiness < burstiness_threshold else "likely not AI-generated"
                verdict_human = "likely not AI-generated" if human_perplexity > perplexity_threshold and human_burstiness < burstiness_threshold else "likely AI-generated"

                results_ai_file.write(f"Sample {total_samples} - Verdict (AI): {verdict_ai}\n")
                results_ai_file.write(f"Analyzed Text (AI): {ai_generated_text}\n\n")
                results_human_file.write(f"Sample {total_samples} - Verdict (Human): {verdict_human}\n")
                results_human_file.write(f"Analyzed Text (Human): {human_text}\n\n")

                if verdict_ai == "likely AI-generated":
                    total_ai_correct += 1
                if verdict_human == "likely not AI-generated":
                    total_human_correct += 1

                ai_accuracy = (total_ai_correct / total_samples) * 100
                human_accuracy = (total_human_correct / total_samples) * 100
                cumulative_accuracy = ((total_ai_correct + total_human_correct) / (total_samples * 2)) * 100

                ai_accuracy_data.append(ai_accuracy)
                human_accuracy_data.append(human_accuracy)
                cumulative_accuracy_data.append(cumulative_accuracy)

        with open("accuracy.txt", "w") as accuracy_file:
            accuracy_file.write(f"Overall Cumulative Accuracy: {cumulative_accuracy:.2f}%\n")
            accuracy_file.write(f"AI Text Detection Accuracy: {ai_accuracy:.2f}%\n")
            accuracy_file.write(f"Human Text Detection Accuracy: {human_accuracy:.2f}%\n")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, total_samples + 1), ai_accuracy_data, label='AI Accuracy', color='blue')
        plt.plot(range(1, total_samples + 1), human_accuracy_data, label='Human Accuracy', color='green')
        plt.plot(range(1, total_samples + 1), cumulative_accuracy_data, label='Cumulative Accuracy', color='red', linestyle='--')
        plt.xlabel("Number of Samples")
        plt.ylabel("Accuracy Percentage")
        plt.title("Model Accuracy Over Time for AI and Human Text Classification")
        plt.legend()
        plt.grid()
        plt.savefig("accuracy_graph.png")
        plt.show()

if __name__ == "__main__":
    main()
