import os
import torch
import argparse
from typing import Optional, Dict
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline, ControlNetModel
from diffusers.utils import load_image

from mask_clipseg import MaskGenerator


# Initialize models at module level
print("Loading SDXL Inpainting models...")

# Load ControlNet model
print("Loading ControlNet for enhanced edge preservation...")
CONTROLNET = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

# Load base SDXL inpainting pipeline
PIPE = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=CONTROLNET,
    torch_dtype=torch.float16,
)
PIPE = PIPE.to("cuda")

# Enable memory optimization
PIPE.enable_model_cpu_offload()
if torch.cuda.is_available():
    PIPE.enable_xformers_memory_efficient_attention()

print("SDXL Inpainting models loaded successfully")


class SDXLInpainter:
    def __init__(
        self, 
        device: str = "cuda", 
        use_controlnet: bool = True,
        lora_paths: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the SDXL Inpainting pipeline
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            use_controlnet (bool): Whether to use ControlNet for better edge preservation
            lora_paths (Dict[str, float]): Dictionary of LoRA paths and their weights
        """
        self.device = device
        self.use_controlnet = use_controlnet
        self.pipe = PIPE
        
        # Load LoRA weights if provided
        if lora_paths:
            print(f"Loading {len(lora_paths)} LoRA models...")
            for lora_path, weight in lora_paths.items():
                # Handle HuggingFace LoRAs vs local paths
                try:
                    if "/" in lora_path and not os.path.exists(lora_path):
                        # Split the path to check for weight file specification
                        if ":" in lora_path:
                            repo_id, weight_name = lora_path.split(":", 1)
                            adapter_name = repo_id.split("/")[-1]
                            print(f"Loading LoRA from HuggingFace: {repo_id} with weight file: {weight_name}")
                            self.pipe.load_lora_weights(
                                repo_id, 
                                weight_name=weight_name,
                                adapter_name=adapter_name
                            )
                        else:
                            # Try to load with default weight name
                            print(f"Loading LoRA from HuggingFace: {lora_path}")
                            print("Specify weight file with repo_id:filename.safetensors if loading fails")
                            adapter_name = lora_path.split("/")[-1]
                            self.pipe.load_lora_weights(
                                lora_path,
                                adapter_name=adapter_name
                            )
                    else:
                        self.pipe.load_lora_weights(lora_path)
                except ValueError as e:
                    if "more than one weights file" in str(e):
                        print(f"Error: {str(e)}")
                        print("Please specify the weight file using format: repo_id:filename.safetensors")
                        continue
                    else:
                        raise
                print(f"Loaded LoRA from {lora_path} with weight {weight}")
            
            # Set the cross-attention scale for the LoRAs
            self.cross_attention_kwargs = {"scale": weight}
        else:
            self.cross_attention_kwargs = None
        
        # Create output directory
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def set_lora_scale(self, scale: float):
        """Set the LoRA scale for inference
        
        Args:
            scale (float): Scale to apply to all loaded LoRAs
        """
        self.cross_attention_kwargs = {"scale": scale}
    
    def inpaint(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        mask_prompt: Optional[str] = None,
        prompt: str = "detailed realistic image, high resolution, 8k, photorealistic",
        negative_prompt: str = "blurry, distorted, low quality, deformed, ugly",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        strength: float = 0.99,
        output_name: Optional[str] = None,
        verbose: int = 0,
        lora_scale: Optional[float] = None
    ) -> Image.Image:
        """
        Perform inpainting on an image
        
        Args:
            image_path (str): Path to the input image
            mask_path (str, optional): Path to the mask image. If not provided, will use mask_prompt with MaskGenerator
            mask_prompt (str, optional): Prompt for MaskGenerator to create a mask
            prompt (str): Prompt for the inpainting
            negative_prompt (str): Negative prompt for the inpainting
            guidance_scale (float): Guidance scale for the inpainting
            num_inference_steps (int): Number of inference steps
            strength (float): Strength of the inpainting
            output_name (str, optional): Name of the output file. If not provided, will use the input name with '_inpainted' suffix
            verbose (int): Whether to display plots (0=no, 1=yes)
            
        Returns:
            PIL.Image: Inpainted image
        """
        # Load the input image
        input_image = load_image(image_path).convert("RGB")
        
        # Get or generate the mask
        if mask_path:
            mask_image = load_image(mask_path).convert("L")
            print(f"Using provided mask: {mask_path}")
        elif mask_prompt:
            print(f"Generating mask with prompt: '{mask_prompt}'")
            mask_gen = MaskGenerator(image_path, verbose=verbose)
            mask_image = mask_gen.generate_mask(mask_prompt)
            mask_path = mask_gen.output_path
        else:
            raise ValueError("Either mask_path or mask_prompt must be provided")
        
        # Ensure input image and mask are the same size
        if input_image.size != mask_image.size:
            mask_image = mask_image.resize(input_image.size, Image.LANCZOS)
        
        # Set LoRA scale if provided
        if lora_scale is not None:
            self.set_lora_scale(lora_scale)
            
        # Generate inpainted image
        print(f"Performing inpainting with {num_inference_steps} steps...")
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            cross_attention_kwargs=self.cross_attention_kwargs
        ).images[0]
        
        # Save output
        if output_name is None:
            base_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_name = f"{name_without_ext}_inpainted.png"
        
        output_path = os.path.join(self.output_dir, output_name)
        output.save(output_path)
        print(f"Inpainted image saved to {output_path}")
        
        return output


def main():
    parser = argparse.ArgumentParser(description="Inpaint images using SDXL")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask", type=str, help="Path to mask image (optional)")
    parser.add_argument("--mask_prompt", type=str, help="Prompt for generating mask (if mask not provided)")
    parser.add_argument("--prompt", type=str, default="detailed realistic image, high resolution, 8k, photorealistic", 
                        help="Prompt for inpainting")
    parser.add_argument("--negative_prompt", type=str, default="blurry, distorted, low quality, deformed, ugly", 
                        help="Negative prompt for inpainting")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--strength", type=float, default=0.99, help="Inpainting strength")
    parser.add_argument("--output", type=str, help="Output filename")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose mode (0=no, 1=yes)")
    parser.add_argument("--controlnet", action="store_true", default=True, help="Use ControlNet for better edges")
    parser.add_argument("--lora", type=str, help="Path to LoRA model (comma-separated for multiple). For HuggingFace models, use 'repo_id:filename.safetensors' format to specify the weight file.")
    parser.add_argument("--lora_weight", type=str, default="0.8", help="Weight for LoRA models (comma-separated for multiple, matching --lora order)")
    parser.add_argument("--lora_scale", type=float, help="Scale to apply during inference (overrides weights specified in --lora_weight)")
    
    args = parser.parse_args()
    
    # Parse LoRA paths and weights
    lora_paths = None
    if args.lora:
        lora_paths = {}
        loras = args.lora.split(",")
        weights = args.lora_weight.split(",")
        
        # If only one weight is provided, apply it to all LoRAs
        if len(weights) == 1 and len(loras) > 1:
            weights = [weights[0]] * len(loras)
            
        # Ensure weights match the number of LoRAs
        if len(weights) != len(loras):
            print(f"Warning: Number of weights ({len(weights)}) doesn't match number of LoRAs ({len(loras)}). Using default weight 0.8 for all.")
            weights = ["0.8"] * len(loras)
            
        for i, lora in enumerate(loras):
            lora_paths[lora.strip()] = float(weights[i].strip())
    
    # Initialize inpainter
    inpainter = SDXLInpainter(
        use_controlnet=args.controlnet,
        lora_paths=lora_paths
    )
    
    # Perform inpainting
    inpainter.inpaint(
        image_path=args.image,
        mask_path=args.mask,
        mask_prompt=args.mask_prompt,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        strength=args.strength,
        output_name=args.output,
        verbose=args.verbose,
        lora_scale=args.lora_scale
    )


if __name__ == "__main__":
    main()