from huggingface_hub import list_repo_files

def check_loras(repo_id):
    """Check available weight files in a Hugging Face LoRA repository"""
    try:
        files = list_repo_files(repo_id)
        safetensors_files = [f for f in files if f.endswith('.safetensors')]
        
        print(f"Repository: {repo_id}")
        if safetensors_files:
            print("Available .safetensors weight files:")
            for file in safetensors_files:
                print(f"  - {file}")
        else:
            print("No .safetensors files found.")
        
        bin_files = [f for f in files if f.endswith('.bin')]
        if bin_files:
            print("Available .bin weight files:")
            for file in bin_files:
                print(f"  - {file}")
        
        if not safetensors_files and not bin_files:
            print("No weight files found.")
    except Exception as e:
        print(f"Error: {e}")

# Check specific LoRA repositories
lora_repos = [
    "ostris/photorealistic-slider-sdxl-lora",
    "ntc-ai/SDXL-LoRA-slider.captivating-eyes",
    "ntc-ai/SDXL-LoRA-slider.eyes",

    # "prithivMLmods/Canopus-LoRA-Flux-FaceRealism",
    # "igorgomes3/Skin_Realistic_LoRa",
    # "ehristoforu/PhotoVision-XL-RealisticVision",
    # "sayakpaul/sd-model-finetuned-lora-t4",
    # "SumioWinters/RealisticVisionV60XL_lora",
    # "latent-consistency/lcm-lora-sdxl"  # Fast inference
]

print("Checking LoRA repositories for weight files...\n")
for repo in lora_repos:
    check_loras(repo)
    print()