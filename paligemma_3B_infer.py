from huggingface_hub import login

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# model_id = "VishnuPJ/Paligemma-Caption-MalayaLLM"
# model_id = "VishnuPJ/MalayaLLM-Paligemma-VQA-3B-Full-Precision"
# model_id = "VishnuPJ/MalayaLLM-Paligemma-Caption-3B-Full-Precision"
model_id = "Merged_Paligemma_vpj_Finetune"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to("cuda")
processor = AutoProcessor.from_pretrained(model_id)

from PIL import Image
import requests

# image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png?download=true"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
raw_image = Image.open("Image.png")

while(True):
    # prompt = "ചിത്രം വിശകലനം ചെയ്ത് പ്രധാന ഘടകങ്ങളെ ഉൾക്കൊള്ളുന്ന ഒരു വിവരണം നൽകുക.പ്രധാന വിഷയം, പശ്ചാത്തല വിശദാംശങ്ങൾ, വർണ്ണങ്ങൾ, പ്രവർത്തനങ്ങൾ, വികാരങ്ങൾ എന്നിവയിൽ ശ്രദ്ധ കേന്ദ്രീകരിക്കുക. വിവരണം സരളവും, സമഗ്രവും , ദൃശ്യത്തിന്റെ കൃത്യമായ വിശദീകരണം നൽകുന്നതുമായിരിക്കണം."
    prompt = input("Enter prompt")
    inputs = processor(prompt, raw_image.convert("RGB"), return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=200)
    
    print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
