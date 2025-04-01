# -*- coding: utf-8 -*-
"""Prompt_based_Image_In_Painting_powered_by_ClipSeg.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HIvkOhQ5UM7W357aLLA-_79u2HVLXWAf
"""

! git lfs install
! git clone https://github.com/timojl/clipseg
! pip install diffusers -q
! pip install transformers -q -UU ftfy gradio
! pip install git+https://github.com/openai/CLIP.git -q

from huggingface_hub import notebook_login

notebook_login()

# Commented out IPython magic to ensure Python compatibility.
# %cd clipseg
! ls

import torch
import requests
import cv2
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

from io import BytesIO
from torch import autocast
import requests
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline
from diffusers.schedulers import LMSDiscreteScheduler # Import LMSDiscreteScheduler

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval();

# !pwd
# !ls -alht
! wget https://huggingface.co/spaces/multimodalart/stable-diffusion-inpainting/resolve/main/clipseg/weights/rd64-uni.pth
!mkdir -p weights; mv rd64-uni.pth weights/
# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('/content/clipseg/weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False);

# or load from URL...
image_url = 'https://okmagazine.ge/wp-content/uploads/2021/04/00-promo-rob-pattison-1024x1024.jpg' #'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fimage.tmdb.org%2Ft%2Fp%2Foriginal%2F72xYNWRTVMDiKVa6SVu6EY0S9Np.jpg' #'https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png'
input_image = Image.open(requests.get(image_url, stream=True).raw)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((512, 512)),
])
img = transform(input_image).unsqueeze(0)

# Commented out IPython magic to ensure Python compatibility.
# %cd ..

input_image.convert("RGB").resize((512, 512)).save("init_image.png", "PNG")
plt.imshow(input_image, interpolation='nearest')
plt.show()

prompts = ['face']

# predict
with torch.no_grad():
    preds = model(img.repeat(len(prompts),1,1,1), prompts)[0]

# visualize prediction
_, ax = plt.subplots(1, 5, figsize=(15, 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(input_image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))];
[ax[i+1].text(0, -15, prompts[i]) for i in range(len(prompts))];

filename = f"mask.png"
plt.imsave(filename,torch.sigmoid(preds[0][0]))

img2 = cv2.imread(filename)
gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

(thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

# For debugging only:
cv2.imwrite(filename,bw_image)

# fix color format
cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

Image.fromarray(bw_image)

# Load base inpainting model first
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="main",
    torch_dtype=torch.float16,
    scheduler=LMSDiscreteScheduler.from_config(
        "runwayml/stable-diffusion-inpainting", subfolder="scheduler"
    ),
    use_auth_token=True
)

# Import LoRA loading utilities
from diffusers import PipelineUtilsClass

# Load LoRA weights - using a realistic face imperfections LoRA
lora_model_id = "sayakpaul/sd-model-finetuned-lora-t4"
lora_filename = "pytorch_lora_weights.bin"  # Using .bin file instead of .safetensors

# LoRA loading scale - adjust as needed (0.0 to 1.0)
lora_scale = 0.7

# Load and apply LoRA weights
pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
pipe = pipe.to("cuda")  # Move to CUDA after loading LoRA

# Load images and preprocess them
init_image = Image.open('init_image.png').convert("RGB").resize((512, 512))
mask = Image.open('mask.png').convert("L").resize((512, 512))

# with autocast("cuda"):
#     images = pipe(prompt="muslim with pimples", init_image=init_image_fixed, mask_image=mask_fixed, strength=0.8)["sample"]

# Run the pipeline with specific data types and apply LoRA
with torch.autocast("cuda"):
    images = pipe(
        prompt="realistic portrait with skin imperfections and pores",
        image=init_image,
        mask_image=mask,
        strength=0.6,  # Increased strength for more noticeable effect
        num_inference_steps=50,  # More steps for better quality
        guidance_scale=7.5,      # Standard guidance scale
        cross_attention_kwargs={"scale": lora_scale}  # Apply LoRA scale
    )["images"]

plt.imshow(input_image, interpolation='nearest')
plt.show()
images[0]