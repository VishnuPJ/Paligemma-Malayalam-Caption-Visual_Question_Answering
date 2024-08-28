import argparse
from datasets import load_dataset
from PIL import Image
import torch
from IndicTransTokenizer import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset
import time
from datetime import timedelta
import gc

class Translate_to_mlm():
    def __init__(self) -> None:
        self.ip = IndicProcessor(inference=True)
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B", trust_remote_code=True).to("cuda:0")
        
    def translate(self, texts):
        # `texts` is expected to be a list of sentences.
        batch = self.ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang="mal_Mlym", show_progress_bar=False)
        batch = self.tokenizer(batch, padding="longest", truncation=True, max_length=256, return_tensors="pt").to("cuda:0")

        with torch.inference_mode():
            outputs = self.model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

        with self.tokenizer.as_target_tokenizer():
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        outputs = self.ip.postprocess_batch(outputs, lang="mal_Mlym")
        return outputs

def create_translated_dataset(dataset, save_path, chunk_size=1000, batch_size=32,val):
    new_data = {
        'image': [],
        'text': []
    }
    
    total_files = dataset.num_rows
    start_time = time.time()
    chunk_counter = 0
    
    for i in range(0, total_files, batch_size):
        # Select a batch from the dataset
        batch_indices = range(i, min(i + batch_size, total_files))
        batch = dataset.select(batch_indices)
        
        images = []
        captions = []
        
        for example in batch:
            images.append(example['image'])
            captions.append(example['caption'])
        
        # Translate captions in batches
        translated_captions = traslate_text.translate(captions)
        
        new_data['image'].extend(images)
        new_data['text'].extend(translated_captions)
        
        # Save chunk to disk
        if (i + batch_size) % chunk_size == 0 or (i + batch_size) >= total_files:
            chunk_dataset = Dataset.from_dict(new_data)
            chunk_dataset.save_to_disk(f"{save_path}_chunk_{val}_{chunk_counter}")
            chunk_counter += 1
            
            # Clear the in-memory data and force garbage collection
            del new_data
            new_data = {
                'image': [],
                'text': []
            }
            gc.collect()
        
        elapsed_time = time.time() - start_time
        files_processed = i + batch_size
        files_remaining = total_files - files_processed
        
        if files_processed > 0:
            avg_time_per_file = elapsed_time / files_processed
            estimated_time_remaining = avg_time_per_file * files_remaining
            eta = timedelta(seconds=int(estimated_time_remaining))
            
            print(f"Completed: {files_processed}/{total_files} files. Remaining: {files_remaining} files. ETA: {eta}", end='\r')
        else:
            print(f"Completed: {files_processed}/{total_files} files. Remaining: {files_remaining} files.", end='\r')

    print("\nDataset creation completed.")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Translate dataset text to Malayalam and save the result.")
    parser.add_argument('--save_path', type=str, required=False, help="Path to save the translated dataset.", default="./translated_data")
    parser.add_argument('--chunk_size', type=int, default=1000, help="Number of samples to process in each chunk.")
    parser.add_argument('--batch_size', type=int, default=32, help="Number of samples to process in each batch.")
    args = parser.parse_args()

    traslate_text = Translate_to_mlm()

    # Load the dataset from the provided path
    dataset = load_dataset("unography/SAM-LLAVA-20k")

    # Create and save the translated dataset in chunks
    for val in ["train","test"]
        create_translated_dataset(dataset[val], args.save_path, args.chunk_size, args.batch_size,val)
