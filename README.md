# PaliGemma-3B-MalayaLLM

<img src="https://github.com/user-attachments/assets/8e8937a7-fd47-482c-acaf-48efc3c04597" alt="Baby MalayaLLM" width="300" height="auto">

# Introducing the Developer:
Discover the mind behind this model and stay updated on their contributions to the field
https://www.linkedin.com/in/vishnu-prasad-j/

# Model description
This is a PaliGemma-3B based model for Malayalam captioning and Visual Question Answering.

- **Model type:** A 3B PaliGemma-2 finetuned model on Malayalam captions and queries.
- **Language(s):** Malayalam and English
- **Datasets:**
  * [VishnuPJ/SAM-LLAVA-20k-Malayalam-Caption-Pretrain](https://huggingface.co/datasets/VishnuPJ/SAM-LLAVA-20k-Malayalam-Caption-Pretrain)
  * [VishnuPJ/laion-14k-GPT4V-LIVIS-Captions_Malayalam](https://huggingface.co/datasets/VishnuPJ/laion-14k-GPT4V-LIVIS-Captions_Malayalam)
  * [VishnuPJ/Malayalam-VQA](https://huggingface.co/datasets/VishnuPJ/Malayalam-VQA)
- **Caption Model-Full Precisoin:** [VishnuPJ/MalayaLLM-Paligemma-Caption-3B-Full-Precision](https://huggingface.co/VishnuPJ/MalayaLLM-Paligemma-Caption-3B-Full-Precision)
- **Caption 4bit Quant:** [VishnuPJ/MalayaLLM-Paligemma-Caption-3B-4bitQuant](https://huggingface.co/VishnuPJ/MalayaLLM-Paligemma-Caption-3B-4bitQuant)
- **VQA Model-Full Precison:** [VishnuPJ/MalayaLLM-Paligemma-VQA-3B-Full-Precision](https://huggingface.co/VishnuPJ/MalayaLLM-Paligemma-VQA-3B-Full-Precision)
- **VQA 4bit Quant:** [VishnuPJ/MalayaLLM-Paligemma-VQA-3B-4bitQuant](https://huggingface.co/VishnuPJ/MalayaLLM-Paligemma-VQA-3B-4bitQuant)
- **VQA LORA Adapters:** [VishnuPJ/MalayaLLM-Paligemma-VQA-3B-Adapters](https://huggingface.co/VishnuPJ/MalayaLLM-Paligemma-VQA-3B-Adapters)
- **Training Precision:** `float16`,`4bit`

# Dataset Creation
I have used [indictrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) for translating English datasets to Malayalam.

Refer "translate_to_mlm.py"

## 💾 Installation Instructions
* pip -q install -U git+https://github.com/huggingface/transformers.git datasets wandb
* pip -q install trl peft accelerate bitsandbytes

# 🌟Happy coding💻🌟
