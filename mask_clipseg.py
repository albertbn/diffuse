import torch
import os
import cv2
import numpy as np
from typing import Optional
from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt


# Initialize model outside the class
print('Loading CLIPDensePredT model...')
# The CLIP portion of the model automatically downloads pretrained weights to ~/.cache/clip/ directory
MODEL: CLIPDensePredT = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
MODEL.eval()
print('CLIPDensePredT version ViT-B/16 loaded successfully')

# non-strict, because we only stored decoder weights (not CLIP weights)
MODEL.load_state_dict(torch.load('./clipseg/weights/rd64-uni.pth',
                      map_location=torch.device('cuda')), strict=False)
print('clipseg model loaded successfully')


class MaskGenerator:
    def __init__(self, image_path: str, verbose: int = 0):
        """
        Initialize the mask generator with an image path
        
        Args:
            image_path (str): Path to the input image
            verbose (int): Whether to display plots (0=no, 1=yes)
        """
        self.image_path: str = image_path
        self.verbose: int = verbose
        self.bw_image: Optional[Image.Image] = None
        
        # Create output directory if it doesn't exist
        self.output_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Generate output filename
        base_name: str = os.path.basename(image_path)
        name_without_ext: str = os.path.splitext(base_name)[0]
        self.output_path: str = os.path.join(self.output_dir, f"{name_without_ext}_mask.png")
        
        # Set up image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def generate_mask(self, prompt: str) -> Image.Image:
        """
        Generate a mask for the given prompt
        
        Args:
            prompt (str): Text prompt describing what to mask
            
        Returns:
            Image.Image: Black and white mask image
        """
        # Load and transform image
        input_image: Image.Image = Image.open(self.image_path).convert('RGB')
        img: torch.Tensor = self.transform(input_image).unsqueeze(0)
        
        # Generate mask
        with torch.no_grad():
            preds: torch.Tensor = MODEL(img, [prompt])[0]
        
        # Convert to numpy array and apply sigmoid to get probability mask (values between 0 and 1)
        mask_array: np.ndarray = torch.sigmoid(preds[0][0]).cpu().numpy()
        
        # Convert to 8-bit grayscale (0-255)
        gray_mask: np.ndarray = (mask_array * 255).astype(np.uint8)
        
        # Apply threshold directly to create binary mask
        binary_mask: np.ndarray
        _, binary_mask = cv2.threshold(gray_mask, 100, 255, cv2.THRESH_BINARY)
        
        # Save final mask
        cv2.imwrite(self.output_path, binary_mask)
        print(f"Mask saved to {self.output_path}")
        
        # Convert to PIL Image
        self.bw_image = Image.fromarray(binary_mask)

        # Show plot if verbose
        if self.verbose:
            self.display_results(prompt, input_image)
        
        return self.bw_image

    def display_results(self, prompt: str, input_image: Image.Image):

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(self.bw_image, cmap='gray')
        plt.title(f"Mask for '{prompt}'")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


# Example usage (will be executed if running this script directly)
if __name__ == "__main__":
    mask_gen: MaskGenerator = MaskGenerator("iwoman.jpeg", verbose=1)
    mask: Image.Image = mask_gen.generate_mask("face")