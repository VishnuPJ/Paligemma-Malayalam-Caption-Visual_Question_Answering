from datasets import load_from_disk
from huggingface_hub import login
login(token = "HF_Token")

# Specify the path to the directory where the DatasetDict is stored
dataset_path = r'./combined_translated'
# Load the DatasetDict from the specified directory
dataset = load_from_disk(dataset_path)

# Example usage: print the keys of the dataset
print(dataset)
dataset.push_to_hub("VishnuPJ/SAM-LLAVA-20k-Malayalam-Caption-Pretrain")


#To upload a model using Huggingface CLI
'''
huggingface-cli login
huggingface-cli upload VishnuPJ/SAM-LLAVA-20k-Malayalam-Caption-Pretrain ./combined_translated
