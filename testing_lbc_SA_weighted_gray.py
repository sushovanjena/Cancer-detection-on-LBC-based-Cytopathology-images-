import os
import torch
import random
import natsort
# import wandb
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import sftp_utils
import cbam_resnet
torch.random.manual_seed(999) 


class WebPDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transforms.Compose([
        transforms.Resize((962, 962)),  # Resize the image to 962x962 pixels
        transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                
                img_name = os.path.join(sftp_utils.sftp_path, self.data.iloc[idx, 2][1:]) 
                image = cv.imread(img_name)  # Read image using OpenCV
                if image is None:  # Check if image is read correctly
                    raise FileNotFoundError(f"Image not found at {img_name}")
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert color space from BGR to RGB
                image = Image.fromarray(image)  # Convert the NumPy array to a PIL Image

                # label = torch.tensor([1.0, 0.0]) if self.data.iloc[idx, 1] == 1 else torch.tensor([0.0, 1.0])
                label = torch.tensor(1.0) if self.data.iloc[idx, 1] == 1 else torch.tensor(0.0)

                if self.transform:
                    image = self.transform(image)

                return image, label
            except Exception as e:
                print(f"Skipping index {idx} due to error: {e}")
                idx += 1  # Move to the next index
                if idx >= len(self.data):  # If index goes out of range, raise StopIteration
                    raise StopIteration("Reached end of dataset after skipping faulty data.")

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        
        # Initialize ResNet18 but without the final fully connected layer
        resnet = models.resnet50(pretrained=False)  # Do not load default weights
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last fully connected layer

         # Adding CBAM to the backbone
        # self.cbam = CBAM(in_planes=2048, ratio=16, kernel_size=7)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Flatten Layer
        # self.flatten = nn.Flatten()

        # self.classifier = nn.Sequential(
        #     nn.Linear(2048, 256),    # Adjust depending on your actual feature map size
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),  # Additional dropout layer
        #     nn.Linear(64, 1)
        # )
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)  # 512 is the output of ResNet18's last conv layer
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.cbam(x)  # Apply CBAM
        # print("x_shape_bb", x.shape)
        # x = self.additional_layers(x) 
        # print("x_shape_al", x.shape)
        x = self.spatial_attention(x) * x 
        x = self.avgpool(x)
        x = self.flatten(x)
        # x = self.flatten(x)
        # x = self.classifier(x)
        x = self.fc1(x)   # Adjust depending on your actual feature map size
        x = F.leaky_relu(x)
        # x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        # x = F.dropout(x, 0.5, training=self.training)  # Additional dropout layer
        x = self.fc3(x)
        x = torch.sigmoid(x)     
        return x

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    checkpoint_path = os.path.join(sftp_utils.sftp_path,'media/Data/2d_annot_model_lbc/dataset/checkpoint_folder_lbc_resnet18_SA_weighted_rectfd_2_gray_resnet50/best_model_fold_4.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # test_corrects = 0
    test_total = 0
    test_tp = 0  # True positives
    test_tn = 0  # True negatives
    test_fp = 0  # False positives
    test_fn = 0  # False negatives

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
        
            outputs = model(image)
            probabilities = outputs 
            preds = (probabilities >= 0.5).float()  # Convert probabilities to predictions using 0.5 as threshold
            test_total += label.size(0)
            # test_corrects += torch.sum(preds == label.data)

            # For calculating specificity and sensitivity
            test_tp += torch.sum((preds == 1) & (label == 1)).item()
            test_tn += torch.sum((preds == 0) & (label == 0)).item()
            test_fp += torch.sum((preds == 1) & (label == 0)).item()
            test_fn += torch.sum((preds == 0) & (label == 1)).item()



    test_accuracy = 100 *  (test_tp + test_tn) / test_total
    test_sensitivity = 100 * test_tp / (test_tp + test_fn) if (test_tp + test_fn) != 0 else 0
    test_specificity = 100 * test_tn / (test_tn + test_fp) if (test_tn + test_fp) != 0 else 0
    test_ppv = 100 * test_tp / (test_tp + test_fp) if (test_tp + test_fp) != 0 else 0
    test_npv = 100 * test_tn / (test_tn + test_fn) if (test_tn + test_fn) != 0 else 0

    print(test_tp,test_tn,test_fp,test_fn,test_total)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Sensitivity: {test_sensitivity:.2f}%')
    print(f'Test Specificity: {test_specificity:.2f}%')
    print(f'Test PPV: {test_ppv:.2f}%')
    print(f'Test NPV: {test_npv:.2f}%')

    
    return test_accuracy, test_sensitivity, test_specificity, test_ppv, test_npv



csv_file_path = os.path.join(sftp_utils.sftp_path,"media/Data/2d_annot_model_lbc/dataset/final_test_data_otsu_rectfd_1.csv")

dataset = WebPDataset(csv_file_path)

# Count total positive and negative input images
positive_count = 0
negative_count = 0

for i in range(len(dataset)):
    _, label = dataset[i]
    if label == 1.0:
        positive_count += 1
    else:
        negative_count += 1

print(f"Total positive images: {positive_count}")
print(f"Total negative images: {negative_count}")

# Step 2: Create an instance of the model
model = CustomModel()
model.eval()

# Step 5: Test the model
test_loader = DataLoader(dataset, batch_size=1, shuffle=True)    

 # Test the model
metrics_all_folds = []
test_accuracy, test_sensitivity, test_specificity, test_ppv, test_npv = test_model(model,test_loader)
metrics = { 'Accuracy': test_accuracy, 'Sensitivity': test_sensitivity, 'Specificity': test_specificity, 'test_ppv': test_ppv, 'test_npv': test_npv}
metrics_all_folds.append(metrics)


# Averaging the results
average_metrics = pd.DataFrame(metrics_all_folds).mean()

# Print or save the average metrics
print("Average Metrics Across All Folds:", average_metrics)
metrics_all_folds_df = pd.DataFrame(metrics_all_folds)
metrics_all_folds_df.to_csv(os.path.join(sftp_utils.sftp_path,'media/Data/2d_annot_model_lbc/dataset/test_metrics_lbc_resnet18_SA_new_rectf_2_resnet50_gray.csv'), index=False)

