
# !pip -q install -U git+https://github.com/huggingface/transformers.git datasets
# !pip -q install trl peft accelerate bitsandbytes

"""We will authenticate to access the model using `notebook_login()`."""
import wandb
from PIL import Image
import torch
import random
from datasets import load_dataset,concatenate_datasets
from huggingface_hub import login
from transformers import PaliGemmaForConditionalGeneration
from transformers import TrainingArguments
from transformers import PaliGemmaProcessor
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
device = "cuda"
login("HF_Token")

wandb.login(key="wandb_token")
wandb.init(project="PaliGemma", name = "VQA")

question_prompts = [
    "ചിത്രം വിശകലനം ചെയ്ത് പ്രധാന ഘടകങ്ങളെ ഉൾക്കൊള്ളുന്ന ഒരു വിവരണം നൽകുക.പ്രധാന വിഷയം, പശ്ചാത്തല വിശദാംശങ്ങൾ, വർണ്ണങ്ങൾ, പ്രവർത്തനങ്ങൾ, വികാരങ്ങൾ എന്നിവയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കുക. വിവരണം സരളവും, സമഗ്രവും , ദൃശ്യത്തിന്റെ കൃത്യമായ വിശദീകരണം നൽകുന്നതുമായിരിക്കണം.",
    "ചിത്രം ശ്രദ്ധാപൂർവ്വം നിരീക്ഷിച്ച് ചിത്രത്തിന് ഒരു അടിക്കുറിപ്പ് നൽകുക.",
    "ചിത്രത്തിന്റെ സമഗ്രമായ വിശദീകരണം ദയവായി നൽകുക.",
    "ചിത്രത്തിന്റെ സമഗ്രമായ വിവരണം നൽകാമോ?",
    "ചിത്രത്തെ വിശദമായി വിവരിക്കുക.",
    "ചിത്രത്തെ അതിന്റെ എല്ലാ വശങ്ങളും ഉൾപ്പെടുത്തി വിശദമായി വിവരിക്കുക.",
    "താഴെ കാണുന്ന ചിത്രത്തിന്റെ വിശദമായ വിവരണം നൽകുക.",
    "ഈ ചിത്രത്തിന്റെ ഉള്ളടക്കം സവിശേഷമായി വിശദീകരിക്കാമോ?",
    "താഴെ കാണുന്ന ചിത്രത്തിന്റെ ഘടകങ്ങളെ വിശദീകരിക്കാമോ.",
    "ചിത്രം വിശകലനം ചെയ്ത് വിശദമായ വിശദീകരണം നൽകാമോ?",
    "ചിത്രത്തെ വിശകലനം ചെയ്യുക, വിശദമായ വിവരണം നൽകുക."
]

def add_random_question(example):
    example["question"] = random.choice(question_prompts)
    return example
    
"""Let's load the dataset."""
ds1 = load_dataset("VishnuPJ/laion-14k-GPT4V-LIVIS-Captions_Malayalam", trust_remote_code=True)
ds2 = load_dataset("VishnuPJ/Malayalam-VQA")

ds2 = ds2.rename_column("multiple_choice_answer", "text")
ds1_with_question = ds1.map(add_random_question, batched=False)
combined_train = concatenate_datasets([ds1_with_question['train'], ds2['train']])
combined_all = concatenate_datasets([combined_train, ds2['train']]).shuffle()

del ds1
del ds2
del ds1_with_question
del combined_train

"""Load the processor to preprocess the dataset."""

model_id = "VishnuPJ/MalayaLLM-Paligemma-Caption-3B-Full-Precision"
processor = PaliGemmaProcessor.from_pretrained(model_id)
"""We will preprocess our examples. We need to prepare a prompt template and pass the text input inside, pass it with batches of images to processor. Then we will set the pad tokens and image tokens to -100 to let the model ignore them. We will pass our preprocessed input as labels to make the model learn how to generate responses."""
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
def collate_fn(examples):
  texts = ["answer " + example["question"] for example in examples]
  labels= [example['text'] for example in examples]
  images = [example['image'].convert("RGB") for example in examples] 
  # images = [example["image"] for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest",
                    tokenize_newline_separately=False)

  tokens = tokens.to(torch.bfloat16).to(device)
  return tokens

"""Our dataset is a very general one and similar to many datasets that PaliGemma was trained with. In this case, we do not need to fine-tune the image encoder, the multimodal projector but we will only fine-tune the text decoder."""
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = True

for param in model.multi_modal_projector.parameters():
    param.requires_grad = True

"""Alternatively, if you want to do LoRA and QLoRA fine-tuning, you can run below cells to load the adapter either in full precision or quantized."""
# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_type=torch.bfloat16
# )

# lora_config = LoraConfig(
#     r=8,
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
#                     "gate_proj", "up_proj", "down_proj"],
#     task_type="CAUSAL_LM",
# )
# model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,
#                                                           quantization_config=bnb_config,
#                                                           device_map={"":0}
#                                                         #   device_map="cpu"
#                                                         )
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()
#trainable params: 11,298,816 || all params: 2,934,634,224 || trainable%: 0.38501616002417344

"""We will now initialize the `TrainingArguments`."""

args=TrainingArguments(
            num_train_epochs=5,
            # max_steps = 100,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=20,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=1,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            # push_to_hub=True,
            save_total_limit=1,
            output_dir="paligemma_vqav2",
            bf16=True,
            # report_to=["tensorboard"],
            report_to="wandb",
            dataloader_pin_memory=False
        )

"""We can now start training."""

from transformers import Trainer

trainer = Trainer(
        model=model,
        train_dataset=combined_all ,
        data_collator=collate_fn,
        args=args
        )



trainer.train()

# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("Merged_Paligemma_vpj_Finetune", save_method = "merged_16bit",)
# processor.save_pretrained("Merged_Paligemma_vpj_Finetune", save_method = "merged_16bit",)

model.save_pretrained("Merged_Paligemma_vpj_Finetune", save_method = "merged_16bit",)
processor.save_pretrained("Merged_Paligemma_vpj_Finetune", save_method = "merged_16bit",)