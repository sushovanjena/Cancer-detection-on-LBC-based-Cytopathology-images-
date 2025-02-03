import torch
import torch.nn as nn
import timm
import os
import numpy as np
import cv2 as cv
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Device Configuration (Use GPU if available, else fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Tile Encoder Model (Pretrained Model for Feature Extraction)
tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
tile_encoder.eval().to(device)

# Define Linear Probe Model (For Classification)
class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int = 1536, num_classes: int = 2):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Image Preprocessing Function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(962, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(image)

# Dataset Class for Handling Image Tiles
class TileDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # Collect all valid image paths
        self.image_paths = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        ]

    def is_white_tile(self, image_path):
        """Check if the image is mostly white using Otsu's thresholding."""
        image = cv.imread(image_path)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        white_pixel_ratio = np.mean(thresh == 255)
        return white_pixel_ratio > 0.95  # Adjust this threshold if needed
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return preprocess_image(image), image_path, self.is_white_tile(image_path)

# Generate Grad-CAM Heatmap
def generate_gradcam_heatmap(model, image, target_class):
    grad_cam = LayerGradCam(model, model.backbone[-2][-1].conv2)
    attributions = grad_cam.attribute(image.unsqueeze(0).to(device), target=target_class)
    heatmap = attributions.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0) / heatmap.max()
    heatmap = cv.resize(heatmap, (224, 224))
    return heatmap

# Save Original Image if it's a White Tile
def save_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(image_path))
    cv.imwrite(save_path, cv.imread(image_path))
    print(f"Saved white tile: {save_path}")

# Save Heatmap Overlayed on Image
def save_heatmap(image, heatmap, img_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = image.permute(1, 2, 0).cpu().numpy()
    heatmap = cv.applyColorMap(np.uint8(255 * heatmap), cv.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed_img = heatmap * 0.4 + np.float32(image)
    superimposed_img = superimposed_img / superimposed_img.max()
    save_path = os.path.join(output_dir, os.path.basename(img_name))
    cv.imwrite(save_path, cv.cvtColor(np.uint8(255 * superimposed_img), cv.COLOR_RGB2BGR))
    print(f"Saved heatmap: {save_path}")

# Classify Tiles & Generate Heatmaps
def classify_and_generate_heatmaps(main_directory, linear_probe, output_dir):
    dataset = TileDataset(os.path.join(main_directory, 'pyramid', 'tiles_files', '17'))     
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    positive_count, negative_count = 0, 0
    
    for images, paths, is_white_tiles in dataloader:
        images = images.to(device)
        
        for img, path, is_white in zip(images, paths, is_white_tiles):
            if is_white:
                save_image(path, output_dir)  # Save white tile as is
                continue
            
            # Generate embedding and classify
            with torch.no_grad():
                embedding = tile_encoder(img.unsqueeze(0))
                prediction = linear_probe(embedding)
                predicted_label = torch.argmax(prediction, dim=1).item()
            
            # Generate and save heatmap
            heatmap = generate_gradcam_heatmap(tile_encoder, img, predicted_label)
            save_heatmap(img, heatmap, path, output_dir)
            
            if predicted_label == 1:
                positive_count += 1
            else:
                negative_count += 1
    
    confidence_score = positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0
    print(f"Confidence Score: {confidence_score:.2f}")

# Main Function
def heatmap(folder_path, output_dir, checkpoint_path):
    # linear_probe_path = "/home/aindra/Documents/2d_lbc_model/aindra-edge_algorithm-d7dcfcfce37e/cpap_edge_algorithm/src/models/best_model_LBC.pth"
    model = LinearProbe(1536, 2).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # main_directory = "/home/aindra/marketing_cases_cyto/4_AINDRAAS0002C00-2779CS20"
    # output_dir = "./heatmaps"
    classify_and_generate_heatmaps(folder_path, model, output_dir)


