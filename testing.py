import os
import torch
import random
import natsort
import sftp_utils
import cv2 as cv
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler

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
                img_name = os.path.join(sftp_utils.sftp_path,self.data.iloc[idx, 2][1:])
                image = cv.imread(img_name)  # Read image using OpenCV
                if image is None:  # Check if image is read correctly
                    raise FileNotFoundError(f"Image not found at {img_name}")
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert color space from BGR to RGB
                image = Image.fromarray(image)  # Convert the NumPy array to a PIL Image

                label = torch.tensor([1.0, 0.0]) if self.data.iloc[idx, 1] == 1 else torch.tensor([0.0, 1.0])

                if self.transform:
                    image = self.transform(image)

                return image, label
            except Exception as e:
                print(f"Skipping index {idx} due to error: {e}")
                idx += 1  # Move to the next index
                if idx >= len(self.data):  # If index goes out of range, raise StopIteration
                    raise StopIteration("Reached end of dataset after skipping faulty data.")
    
class CustomModel(nn.Module):
    def __init__(self, pre_trained_model_path=None):
        super(CustomModel, self).__init__()
        
        # Initialize ResNet18 but without the final fully connected layer
        resnet = models.resnet18(pretrained=False)  # Do not load default weights
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last fully connected layer and avgpool layer

        # If you had a custom layer at the end of the ResNet18, define it
        # Example: 1x1 Convolution replacing the fully connected layer
        self.transition_layer = nn.Conv2d(512, 1024, kernel_size=1)  # Example transition layer

        # Additional layers on top of the modified backbone
        self.additional_layers = nn.Sequential(
             nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),    # Adjust depending on your actual feature map size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Additional dropout layer
            nn.Linear(64, 2)
        )

        if pre_trained_model_path:
            # Load the state dict
            state_dict = torch.load(pre_trained_model_path)
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.transition_layer(x)
        x = self.additional_layers(x) 
        # print("shape before flatten layer", x.shape)
        x = x.view(x.size(0), -1)  # Flatten the features for the classifier
        # print("shape after flatten layer", x.shape)
        x = self.classifier(x)
        return x


def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    checkpoint_path = os.path.join(sftp_utils.sftp_path,f'media/dataset/annot_cases_cpap/data/checkpoint_folder_cpap_otsu/best_model_fold_4.pth')
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

    criterion = nn.BCEWithLogitsLoss()
    test_loss = 0.0

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
        
            outputs = model(image)
            probabilities = torch.softmax(outputs,dim=1)  # Apply sigmoid to convert to probabilities
            preds = (probabilities >= 0.5).float()  # Convert probabilities to predictions using 0.5 as threshold
            test_total += label.size(0)
            # test_corrects += torch.sum(preds == label.data)

            # For calculating specificity and sensitivity
            test_tp += torch.sum((preds[:, 0] == 1) & (label[:, 0] == 1)).item()
            test_tn += torch.sum((preds[:, 1] == 1) & (label[:, 1] == 1)).item()
            test_fp += torch.sum((preds[:, 0] == 1) & (label[:, 1] == 1)).item()
            test_fn += torch.sum((preds[:, 1] == 1) & (label[:, 0] == 1)).item()


            loss = criterion(outputs, label)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_accuracy = 100 *  (test_tp + test_tn) / test_total
    test_sensitivity = 100 * test_tp / (test_tp + test_fn) if (test_tp + test_fn) != 0 else 0
    test_specificity = 100 * test_tn / (test_tn + test_fp) if (test_tn + test_fp) != 0 else 0
    test_ppv = 100 * test_tp / (test_tp + test_fp) if (test_tp + test_fp) != 0 else 0
    test_npv = 100 * test_tn / (test_tn + test_fn) if (test_tn + test_fn) != 0 else 0
    print(test_tp,test_tn,test_fp,test_fn,test_total)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Sensitivity: {test_sensitivity:.2f}%')
    print(f'Test Specificity: {test_specificity:.2f}%')
    print(f'Test PPV: {test_ppv:.2f}%')
    print(f'Test NPV: {test_npv:.2f}%')

    return test_accuracy, test_sensitivity, test_specificity, test_ppv, test_npv


csv_file_path = os.path.join(sftp_utils.sftp_path,"media/dataset/annot_cases_cpap/data/final_test_data_otsu.csv")

dataset = WebPDataset(csv_file_path)

# Step 2: Create an instance of the model
model = CustomModel()
model.eval()

# Step 5: Test the model
test_loader = DataLoader(dataset, batch_size=16, shuffle=True)    

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
metrics_all_folds_df.to_csv(os.path.join(sftp_utils.sftp_path,'media/dataset/annot_cases_cpap/data/test_metrics.csv'), index=False)

# import sftp_utils
# import os

# folder = '/media/dataset/annot_cases_cpap/data/checkpoint_folder_cpap_otsu'
# checkpoint_folder = os.path.join(sftp_utils.sftp_path, folder[1:])
# print("sftep", checkpoint_folder)

