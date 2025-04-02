01-04-2025 18:02

https://github.com/timojl/clipseg


```
scp -i "~/.ssh/FrameAINLPServerKey.pem" ubuntu@ec2-34-204-4-209.compute-1.amazonaws.com:projects/diffuse/output/iwoman_mask.png ./
scp -i "~/.ssh/FrameAINLPServerKey.pem" ubuntu@ec2-34-204-4-209.compute-1.amazonaws.com:projects/diffuse/output/iwoman_inpainted.png ./

git config --global user.name "Albert@g5.16xlarge"
git config --global user.email "Albert@g5.16xlarge"

python inpaint.py --image iwoman2.jpeg --mask_prompt "face" --prompt "detailed realistic crying iranian woman, high resolution, 8k, photorealistic"
python inpaint.py --image iwoman.jpeg --mask_prompt "face" --prompt "detailed realistic crying iranian woman, high resolution, 8k, photorealistic"
python inpaint.py --image iwoman.jpeg --strength 0.3 --mask_prompt "face" --prompt "detailed realistic face of crying iranian woman, high resolution, 8k, photorealistic"
python inpaint.py --image iwoman.jpeg --strength 0.3 --mask_prompt "face" --prompt "detailed realistic face with perfect skin and natural eyes, high resolution, 8k, photorealistic" --lora "igorgomes3/Skin_Realistic_LoRa,sayakpaul/sd-model-finetuned-lora-t4" --lora_weight "0.8,0.6" --lora_scale 0.75

python inpaint.py --image iwoman.jpeg --strength 0.3 --mask_prompt "face" --prompt "detailed realistic face of crying iranian woman with perfect skin and natural eyes, high resolution, 8k,
  photorealistic" --lora "SumioWinters/RealisticVisionV60XL_lora:RealisticVisionV60XL.safetensors,latent-consistency/lcm-lora-sdxl:pytorch_lora_weights.safetensors"
   --lora_weight "0.8,0.6" --lora_scale 0.75

python inpaint.py --image iwoman.jpeg --strength 0.5 --mask_prompt "face" --prompt "detailed realistic face of crying iranian woman, high resolution, 8k, photorealistic"

```
