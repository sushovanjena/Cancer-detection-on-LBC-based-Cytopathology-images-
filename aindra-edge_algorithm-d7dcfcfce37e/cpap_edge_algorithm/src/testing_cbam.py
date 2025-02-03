import os
import torch
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt

torch.random.manual_seed(999)



class WebPDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path (str): Path to the folder containing the WebP images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.webp')])  # List all .webp files
        
        self.transform = transforms.Compose([
            transforms.Resize((962, 962)),  # Resize the image to 962x962 pixels
            transforms.ToTensor(),
        ])
    
    def is_white_tile(self, image):
        """Check if the image is mostly white based on Otsu's thresholding."""
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)        
        # Apply Otsu's thresholding
        _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
        # Calculate the ratio of white pixels
        white_pixel_ratio = np.mean(thresh == 255)
        return white_pixel_ratio > 0.95  # Adjust this ratio threshold if needed
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        while True:
            try:
                img_name = os.path.join(self.folder_path, self.image_files[idx])  
                image = cv.imread(img_name)
                if image is None:
                    raise FileNotFoundError(f"Image not found at {img_name}")
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                is_white = self.is_white_tile(image)
                
                # If the tile is not white, proceed with transforming and returning it
                gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
                gray_image = cv.merge([gray_image, gray_image, gray_image]) 
                gray_image = Image.fromarray(gray_image)
                image = Image.fromarray(image)
                # label = torch.tensor(1.0) if self.data.iloc[idx, 1] == 1 else torch.tensor(0.0)
                if self.transform:
                    image = self.transform(image)
                    gray_image = self.transform(gray_image)
                return image, img_name, gray_image, is_white
            except Exception as e:
                print(f"Skipping index {idx} due to error: {e}")
                idx += 1
                if idx >= len(self.data):
                    raise StopIteration("Reached end of dataset after skipping faulty data.")


# SAM and CAM implementations
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # self.channel_attention = ChannelGate(in_planes, ratio)
        self.spatial_attention = SpatialGate()

    def forward(self, x):
        # out = self.channel_attention(x)
        out = self.spatial_attention(x)
        return out


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Initialize ResNet18 but without the final fully connected layer
        # resnet = cbam_resnet.resnet50_cbam(pretrained=False)  # Do not load default weights
        resnet = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last fully connected layer and avgpool layer

         # Adding CBAM to the backbone
        self.cbam = CBAM(in_planes=2048, ratio=16, kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        # Flatten Layer
        self.flatten = nn.Flatten()

        
        self.fc1 = nn.Linear(2048, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)  # Apply CBAM
        # print("x_shape_bb", x.shape)
        # x = self.additional_layers(x) 
        # print("x_shape_al", x.shape)
        x = self.avgpool(x)
        x = self.flatten(x)
        # x = self.classifier(x)
        x = self.fc1(x)   # Adjust depending on your actual feature map size
        x = F.leaky_relu(x)
        # x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        # x = F.dropout(x, 0.5, training=self.training)  # Additional dropout layer
        x = self.fc3(x)    
        return x

    
# Apply Grad-CAM
def generate_gradcam_heatmap(model, image, target_class):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # print(model.backbone[-2][-1].conv3)
    # image = image.unsqueeze(0) 
    image = image.to(device)
    print("ms",image.shape)
    model.eval()
    grad_cam = LayerGradCam(model, model.backbone[-2][-1].conv2)
    # print(target_class)
     # Ensure target_class is valid for binary classification
    if target_class not in [0, 1]:
        raise ValueError("target_class must be 0 or 1 for binary classification")

    # Since model output shape is [1, 1], it implies a single logit or probability
 
    # For binary classification, Grad-CAM usually visualizes the class of interest
    # Here we assume that the model output is a single score per image
    if target_class == 1:
        target = torch.tensor([target_class]).to(device)  # Index for positive class
        # print('target',target)
    # else:
    #     target = torch.tensor([0])  # Index for negative class
    # else:
    #     target = torch.tensor([target_class])

    # Generate Grad-CAM heatmap
    attributions = grad_cam.attribute(image, target=target)
    

    # attributions = grad_cam.attribute(image, target=1)
    print(f"GradCam attributions shape: {attributions.shape}")
    heatmap = attributions.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / heatmap.max()
    heatmap = cv.resize(heatmap, (image.size(-1), image.size(-2)))

    return heatmap


def visualize_heatmap_on_image(image, heatmap):
    image = image.permute(1, 2, 0).cpu().numpy()
    heatmap = cv.applyColorMap(np.uint8(255 * heatmap), cv.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    superimposed_img = heatmap * 0.4 + np.float32(image)
    superimposed_img = superimposed_img / superimposed_img.max()
    return superimposed_img

def save_heatmap(heatmap_image, img_name, output_dir):

    if isinstance(img_name, tuple):
        img_name = img_name[0]  # Assuming the actual path is the first element in the tuple

    # Extract original image name and extension
    base_name = os.path.basename(img_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the heatmap with the same name and format
    heatmap_path = os.path.join(output_dir, base_name)
    cv.imwrite(heatmap_path, cv.cvtColor(np.uint8(255 * heatmap_image), cv.COLOR_RGB2BGR))
    print(f"Saved heatmap: {heatmap_path}")

def save_normal_image(image_tensor, img_name, output_dir):
    """
    Saves the normal image (without heatmap) from a tensor.
    
    Args:
        image_tensor: A torch tensor representing the image.
        img_name: The name of the file to save the image as.
    """
    if isinstance(img_name, tuple):
        img_name = img_name[0]  # Assuming the actual path is the first element in the tuple

    # Extract original image name and extension
    base_name = os.path.basename(img_name)
    os.makedirs(output_dir, exist_ok=True)

    image_tensor = torch.clamp(image_tensor, 0, 1)  # Ensure values are between 0 and 1
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # Convert to HxWxC
    
    # Save the image using OpenCV
    normal_image_path = os.path.join(output_dir, base_name)
    cv.imwrite(normal_image_path, cv.cvtColor(np.uint8(255 * image_np), cv.COLOR_RGB2BGR))
    
    print(f"Saved normal image: {normal_image_path}")


def test_model(model, checkpoint_path, test_loader, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    # checkpoint_path = os.path.join(sftp_utils.sftp_path, 'media/dataset/annot_cases_cpap/sipakmed_dataset/cervical-cancer-largest-dataset-sipakmed/checkpoint_folder_cpap_resnet50_endcbam_2_finetune/best_model.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # test_total, test_tp, test_tn, test_fp, test_fn = 0, 0, 0, 0, 0

    with torch.no_grad():
        for batch_idx, (image, img_name, gray_image, is_white) in enumerate(test_loader):
            image = image.to(device)
            total_images = 0
            positive_images = 0


            # Process each image in the batch individually
            for i in range(image.size(0)):
                total_images += 1  # Increment total images for each processed image
                if not is_white[i]:  # Check if this specific tile is not white
                    output = model(image[i].unsqueeze(0))  # Forward pass for a single image
                    print(f"Model output shape: {output}")
                    prediction = F.softmax(output,dim=1)
                    pred = torch.argmax(prediction,dim=1)
                    print(f"Model preds shape: {pred.item()}")
                    if pred.item() == 1:
                        positive_images += 1  # Increment positive count if prediction is positive
                    # test_total += 1
                    # test_tp += ((pred == 1) & (label[i] == 1)).item()
                    # test_tn += ((pred == 0) & (label[i] == 0)).item()
                    # test_fp += ((pred == 1) & (label[i] == 0)).item()
                    # test_fn += ((pred == 0) & (label[i] == 1)).item()

                    # Generate Grad-CAM and save heatmap for non-white images
                    gradcam_heatmap = generate_gradcam_heatmap(model, image[i].unsqueeze(0), target_class=1)
                    superimposed_img = visualize_heatmap_on_image(image[i].cpu(), gradcam_heatmap)
                    save_heatmap(superimposed_img, img_name[i], output_dir)
                else:
                    # Directly save white tile without Grad-CAM
                    save_normal_image(image[i].cpu(), img_name[i], output_dir)

     # Calculate the percentage of positive images
    positive_percentage = 100 * positive_images / total_images
    if positive_percentage > 30:
        classification_result = "Abnormal"
    else:
        classification_result = "Normal"

    return classification_result
    # test_accuracy = 100 * (test_tp + test_tn) / test_total
    # test_sensitivity = 100 * test_tp / (test_tp + test_fn) if (test_tp + test_fn) != 0 else 0
    # test_specificity = 100 * test_tn / (test_tn + test_fp) if (test_tn + test_fp) != 0 else 0
    # test_ppv = 100 * test_tp / (test_tp + test_fp) if (test_tp + test_fp) != 0 else 0
    # test_npv = 100 * test_tn / (test_tn + test_fn) if (test_tn + test_fn) != 0 else 0

    # print(test_tp, test_tn, test_fp, test_fn, test_total)
    # print(f'Test Accuracy: {test_accuracy:.2f}%')
    # print(f'Test Sensitivity: {test_sensitivity:.2f}%')
    # print(f'Test Specificity: {test_specificity:.2f}%')
    # print(f'Test PPV: {test_ppv:.2f}%')
    # print(f'Test NPV: {test_npv:.2f}%')






def heatmap(folder_path, output_dir, checkpoint_path):
    # Create dataset and loader
    dataset = WebPDataset(folder_path)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create an instance of the model
    model = CustomModel()

    classification_result = test_model(model,checkpoint_path,test_loader,output_dir)

    return classification_result

