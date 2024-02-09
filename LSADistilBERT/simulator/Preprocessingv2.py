import os
import json
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

# Initialize the DistilBertTokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')

max_seq_length = 256

# Define the context variable and pair each token with the word it represents
with open("LSADistilBERT\simulator\Dataset.json") as f:
    data = json.load(f)
encoded_data = []
# Modify preprocessing code to handle multiple questions per example
for example in data:
    context = example["context"]
    for qas in example["qas"]:
        question = qas["question"]

        # Create a separate encoded example for each question
        encoded_example = {}
        encoded_example["context"] = distilbert_tokenizer.encode(context, add_special_tokens=True, max_length=max_seq_length, truncation=True)
        encoded_example["question"] = distilbert_tokenizer.encode("[Q] " + question + " [C] " + context, add_special_tokens=True, max_length=max_seq_length, truncation=True)
        encoded_example["start_position"] = qas["answers"][0]["answer_start"]
        encoded_example["end_position"] = qas["answers"][0]["answer_start"] + len(qas["answers"][0]["text"]) - 1
        encoded_data.append(encoded_example)


# Split the data into train and test sets (70% train, 30% test)
train_data, test_data = train_test_split(encoded_data, test_size=0.3, random_state=42)

# Define the path to the folder where you want to save the JSON files
folder_path = "LSADistilBERT/simulator"

# Define the file paths for the JSON files
train_file_path = f"{folder_path}/train.json"
test_file_path = f"{folder_path}/test.json"

# Save the preprocessed train and test sets to JSON files
with open(train_file_path, "w") as f:
    json.dump(train_data, f)
with open(test_file_path, "w") as f:
    json.dump(test_data, f)