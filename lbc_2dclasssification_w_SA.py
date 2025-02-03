import os
import gc
import torch
import random
import natsort
# import wandb
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
import sftp_utils
import cbam_resnet
torch.random.manual_seed(999) 

torch.cuda.empty_cache()

class WebPDataset(Dataset):
    def __init__(self, csv_file, transform=None,  with_augmentation = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = pd.read_csv(csv_file)
        self.transform_wo_aug = transforms.Compose([  
        transforms.Resize((962, 962)),  # Resize the image to 962x962 pixels
        transforms.ToTensor(),
        ])

        self.transform_w_aug = transforms.Compose([
            transforms.Resize((962, 962)),  # Resize the image to 962x962 pixels
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with a 50% probability
            transforms.RandomRotation(degrees=15),  # Random rotation within 15 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random changes in brightness, contrast, saturation, and hue
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet means and stds
        ])

        self.with_augmentation = with_augmentation

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

                # image = self.transform_wo_aug(image)
                
                if self.with_augmentation:
                    image = self.transform_w_aug(image)
                else:
                    image = self.transform_wo_aug(image)

                return image, label
            except Exception as e:
                print(f"Skipping index {idx} due to error: {e}")
                idx += 1  # Move to the next index
                if idx >= len(self.data):  # If index goes out of range, raise StopIteration
                    raise StopIteration("Reached end of dataset after skipping faulty data.")


# # Your provided SAM and CAM implementations
# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()
#         kernel_size = 7
#         self.compress = ChannelPool()
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = torch.sigmoid(x_out)  # broadcasting
#         return x * scale

# def logsumexp_2d(tensor):
#     tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
#     s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
#     outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
#     return outputs

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super(ChannelGate, self).__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#         )
#         self.pool_types = pool_types

#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type == 'avg':
#                 avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp(avg_pool)
#             elif pool_type == 'max':
#                 max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp(max_pool)
#             elif pool_type == 'lp':
#                 lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#                 channel_att_raw = self.mlp(lp_pool)
#             elif pool_type == 'lse':
#                 lse_pool = logsumexp_2d(x)
#                 channel_att_raw = self.mlp(lse_pool)

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum = channel_att_sum + channel_att_raw

#         scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         return x * scale

# class CBAM(nn.Module):
#     def __init__(self, in_planes, ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelGate(in_planes, ratio)
#         self.spatial_attention = SpatialGate()

#     def forward(self, x):
#         out = self.channel_attention(x)
#         out = self.spatial_attention(out)
#         return out
    
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
        resnet = models.resnet18(pretrained=True)  # Do not load default weights
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last fully connected layer and avgpool layer

         # Adding CBAM to the backbone
        # self.cbam = CBAM(in_planes=2048, ratio=16, kernel_size=7)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Flatten Layer
        self.flatten = nn.Flatten()

        # self.classifier = nn.Sequential(
        #     nn.Linear(2048, 256),    # Adjust depending on your actual feature map size
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 64),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.5),  # Additional dropout layer
        #     nn.Linear(64, 1)
        # )
        # self.fc1 = nn.Linear(2048, 256)
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, 1)
        self.fc1 = nn.Linear(512, 256)  # 512 is the output of ResNet18's last conv layer
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.cbam(x)  # Apply CBAM
        # print("x_shape_bb", x.shape)
        # x = self.additional_layers(x) 
        # print("x_shape_al", x.shape)
        x = self.spatial_attention(x) * x 
        x = self.avgpool(x)
        x = self.flatten(x)
        # x = self.classifier(x)
        x = self.fc1(x)   # Adjust depending on your actual feature map size
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        # print('x_shape_after fc2', x.shape)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.5, training=self.training)  # Additional dropout layer
        x = self.fc3(x)
        # print('x_shape_after fc3', x.shape)
        x = torch.sigmoid(x)     
        # print('x_shape_after sigmoid', x)
        return x




class CancerClassifier:
    def __init__(self, dataset, model):
        self.model = model
        self.dataset = dataset
        
    
    def split_data(self, dataset, train_ratio=0.8):
        """
        Splits the dataset into training and validation sets with equal number of positive and negative examples in the validation set.

        Parameters:
        - dataset (Dataset): The dataset to split.
        - train_ratio (float): The proportion of each category (positive and negative) to include in the train split.

        Returns:
        - train_dataset, val_dataset (Dataset, Dataset): The training and validation datasets.
        """
        # positive_indices = [i for i, (_, label) in enumerate(dataset) if label[0].item() == 1.0 and label[1].item() == 0.0]
        # negative_indices = [i for i, (_, label) in enumerate(dataset) if label[0].item() == 0.0 and label[1].item() == 1.0]
        
        positive_indices = [i for i, (_, label) in enumerate(dataset) if label.item() == 1.0]
        negative_indices = [i for i, (_, label) in enumerate(dataset) if label.item() == 0.0]


        # Calculate the number of samples for each set
        num_positives = len(positive_indices)
        num_negatives = len(negative_indices)
        min_category_size = min(num_positives, num_negatives)
        train_size_per_category = int(train_ratio * min_category_size)

        # Shuffle indices
        random.shuffle(positive_indices)
        random.shuffle(negative_indices)

        # Split indices for training and validation sets
        train_indices = positive_indices[:train_size_per_category] + negative_indices[:train_size_per_category]
        val_indices = positive_indices[train_size_per_category:min_category_size] + negative_indices[train_size_per_category:min_category_size]

        # Shuffle the combined indices
        random.shuffle(train_indices)
        random.shuffle(val_indices)

        # Create training and validation subsets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        print(len(train_dataset),len(val_dataset))

        return train_dataset, val_dataset


    def train(self, train_loader, val_loader, num_epochs, learning_rate, fold, latest_epoch, checkpoint_folder=None, resume=False, early_stopping_patience=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # scaler = GradScaler()

        # Initialize the learning rate scheduler
        scheduler = StepLR(optimizer, step_size = 10, gamma = 0.1)  # Adjust step_size and gamma as needed

        best_val_loss = float('inf')
        best_epoch = 0

        start_epoch = 0
        epochs_no_improve = 0  # Initialize counter outside the loop
        # if resume:
        #     checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_fold_{fold}_epoch_{latest_epoch}.pth')  
        #     if os.path.isfile(checkpoint_path):
        #         print(f"Resuming training from checkpoint {checkpoint_path}")
        #         checkpoint = torch.load(checkpoint_path,map_location=device)
        #         start_epoch = checkpoint['epoch']
        #         print(start_epoch)
        #         best_val_loss = checkpoint['best_val_loss']
        #         self.model.load_state_dict(checkpoint['state_dict'])
        #         optimizer.load_state_dict(checkpoint['optimizer'])
        #         scaler.load_state_dict(checkpoint['scaler'])

        for epoch in range(start_epoch,num_epochs):
            print("training started")
            self.model.train()
            running_loss = 0.0
            train_total = 0
            # train_corrects = 0
            train_tp = 0  # True positives
            train_tn = 0  # True negatives
            train_fp = 0  # False positives
            train_fn = 0  # False negatives

            for image, label in train_loader:
                image, label = image.to(device), label.to(device)
                # print('features_shape', features.shape,'\n','label shape', label.shape)
                train_total += label.size(0)

                optimizer.zero_grad()

                # with autocast():
                outputs = self.model(image)
                # print('outputs', outputs.shape)
                label = label.unsqueeze(1)
                # print("outputs, label", outputs, label)
                
                # loss = criterion(outputs, label)
                loss1 = criterion(outputs[0], label[0]) #weighted loss
                loss2 = criterion(outputs[1], label[1])
                if label[0] == 1.0:
                    loss1 = 2*loss1
                if label[1] == 1.0:
                    loss2 = 2*loss2
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                running_loss += loss.item()
            

                probabilities = outputs  # Apply sigmoid to convert to probabilities
                preds = (probabilities >= 0.5).float()  # Convert probabilities to predictions using 0.5 as threshold
                # train_corrects = torch.sum(preds == label.data)

                # For calculating specificity and sensitivity
                # train_tp += torch.sum((preds[:, 0] == 1) & (label[:, 0] == 1)).item()
                # train_tn += torch.sum((preds[:, 1] == 1) & (label[:, 1] == 1)).item()
                # train_fp += torch.sum((preds[:, 0] == 1) & (label[:, 1] == 1)).item()
                # train_fn += torch.sum((preds[:, 1] == 1) & (label[:, 0] == 1)).item()
                train_tp += torch.sum((preds == 1) & (label == 1)).item()
                train_tn += torch.sum((preds == 0) & (label == 0)).item()
                train_fp += torch.sum((preds == 1) & (label == 0)).item()
                train_fn += torch.sum((preds == 0) & (label == 1)).item()


            train_accuracy = 100 * (train_tp + train_tn) / train_total
            train_sensitivity = 100 * train_tp / (train_tp + train_fn) if (train_tp + train_fn) != 0 else 0
            train_specificity = 100 * train_tn / (train_tn + train_fp) if (train_tn + train_fp) != 0 else 0
            train_ppv = 100 * train_tp / (train_tp + train_fp) if (train_tp + train_fp) != 0 else 0
            train_npv = 100 * train_tn / (train_tn + train_fn) if (train_tn + train_fn) != 0 else 0
            
            print(f'Epoch {epoch+1} - Train Loss: {running_loss / len(train_loader):.4f}')
            print(f'Epoch {epoch+1} - Train Accuracy: {train_accuracy:.2f}%')
            print(f'Epoch {epoch+1} - Train Sensitivity: {train_sensitivity:.2f}%')
            print(f'Epoch {epoch+1} - Train Specificity: {train_specificity:.2f}%')
            print(f'Epoch {epoch+1} - Train PPV: {train_ppv:.2f}%')
            print(f'Epoch {epoch+1} - Train NPV: {train_npv:.2f}%')

            # wandb.log({"train loss": running_loss / len(train_loader),
            #  "train_accuracy": train_accuracy,
            #  "train_sensitivity": train_sensitivity,
            #  "train_specificity": train_specificity,
            #  "train_ppv": train_ppv,
            #  "train_npv": train_npv,
            #  })


            # Validation
            self.model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_total = 0
            val_tp = 0  # True positives
            val_tn = 0  # True negatives
            val_fp = 0  # False positives
            val_fn = 0  # False negatives

            with torch.no_grad():
                for image, label in val_loader:
                    image, label = image.to(device), label.to(device)

                    outputs = self.model(image)
                    label = label.unsqueeze(1)
                    loss = criterion(outputs, label)
                    val_loss += loss.item()
                    probabilities = outputs
                    preds = (probabilities >= 0.5).float()  # Convert probabilities to predictions using 0.5 as threshold
                    # print("label=",label,"pred=",preds)
                    val_total += label.size(0)
                    # val_corrects = torch.sum(preds == label.data)

                    # For calculating specificity and sensitivity
                    val_tp += torch.sum((preds == 1) & (label == 1)).item()
                    val_tn += torch.sum((preds == 0) & (label == 0)).item()
                    val_fp += torch.sum((preds == 1) & (label == 0)).item()
                    val_fn += torch.sum((preds == 0) & (label == 1)).item()

            val_loss /= len(val_loader)
            val_accuracy = 100 *  (val_tp + val_tn) / val_total
            val_sensitivity = 100 * val_tp / (val_tp + val_fn) if (val_tp + val_fn) != 0 else 0
            val_specificity = 100 * val_tn / (val_tn + val_fp) if (val_tn + val_fp) != 0 else 0
            val_ppv = 100 * val_tp / (val_tp + val_fp) if (val_tp + val_fp) != 0 else 0
            val_npv = 100 * val_tn / (val_tn + val_fn) if (val_tn + val_fn) != 0 else 0

            print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')
            print(f'Epoch {epoch+1} - Validation Accuracy: {val_accuracy:.2f}%')
            print(f'Epoch {epoch+1} - Validation Sensitivity: {val_sensitivity:.2f}%')
            print(f'Epoch {epoch+1} - Validation Specificity: {val_specificity:.2f}%')
            print(f'Epoch {epoch+1} - Validation PPV: {val_ppv:.2f}%')
            print(f'Epoch {epoch+1} - Validation NPV: {val_npv:.2f}%')


            # wandb.log({"val_loss": val_loss,
            #  "val_accuracy": val_accuracy,
            #  "val_sensitivity": val_sensitivity,
            #  "val_specificity": val_specificity,
            #  "val_ppv": val_ppv,
            #  "val_npv": val_npv,
            #  })



             # Save checkpoint
            if checkpoint_folder:
                checkpoint = {
                    'fold': fold,
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(checkpoint_folder, f'checkpoint_fold_{fold}_epoch_{epoch + 1}.pth'))

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0  # Reset counter

                # Save the best model
                best_model_path = os.path.join(checkpoint_folder, f'best_model_fold_{fold}.pth')
                torch.save(self.model.state_dict(), best_model_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f'Early stopping triggered after epoch {epoch + 1}. No improvement in validation loss for {early_stopping_patience} consecutive epochs.')
                    break  # Early stopping

            # Step the scheduler
            scheduler.step()

            torch.cuda.empty_cache()
            gc.collect()

        print('Training complete.')
        print(f'Best model found at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}')

def test_model(model, test_loader, fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    checkpoint_path = os.path.join(sftp_utils.sftp_path,f'media/Data/2d_annot_model_lbc/dataset/checkpoint_folder_lbc_resnet18_SA_rectfd_1/best_model_fold_{fold}.pth')
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

    # criterion = nn.BCELoss()
    # test_loss = 0.0

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
    # print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Sensitivity: {test_sensitivity:.2f}%')
    print(f'Test Specificity: {test_specificity:.2f}%')
    print(f'Test PPV: {test_ppv:.2f}%')
    print(f'Test NPV: {test_npv:.2f}%')

    
    return test_accuracy, test_sensitivity, test_specificity, test_ppv, test_npv


def load_datasets(csv_files, folder):
    """Load datasets from a list of CSV files, both with and without augmentation.
    
    Args:
        csv_files (list of str): List of CSV file names.
        folder (str): The directory containing the CSV files.
    
    Returns:
        ConcatDataset: A concatenated dataset containing both augmented and non-augmented data.
    """
    # Load datasets without augmentation
    datasets_wo_aug = [WebPDataset(os.path.join(folder, csv_file), with_augmentation=False) for csv_file in csv_files]
    dataset_wo_aug = ConcatDataset(datasets_wo_aug)
    
    # Load datasets with augmentation
    datasets_w_aug = [WebPDataset(os.path.join(folder, csv_file), with_augmentation=True) for csv_file in csv_files]
    dataset_w_aug = ConcatDataset(datasets_w_aug)
    
    # Concatenate both the augmented and non-augmented datasets
    combined_dataset = ConcatDataset([dataset_wo_aug, dataset_w_aug])

    return combined_dataset


def k_fold_cross_validation(dataset_folder, num_epochs, learning_rate, checkpoint_folder, resume=False):
    csv_files = natsort.natsorted(os.listdir(dataset_folder))
    print(csv_files)
    metrics_all_folds = []

    start_fold = 1
    # wandb.login()
    # Load the last checkpoint if resume training
    if resume:
        latest_checkpoint_path = None
        latest_epoch = -1
        for file in os.listdir(checkpoint_folder):
            if file.startswith('checkpoint_fold_') and file.endswith('.pth'):
                checkpoint_fold = int(file.split('_')[2])
                file_epoch = file.split('_')[4]
                checkpoint_epoch = int(file_epoch.split('.')[0])
                if checkpoint_fold > start_fold or (checkpoint_fold == start_fold and checkpoint_epoch > latest_epoch):
                    latest_checkpoint_path = os.path.join(checkpoint_folder, file)
                    start_fold = checkpoint_fold
                    latest_epoch = checkpoint_epoch
                    # print(latest_epoch)
        
        if latest_checkpoint_path:
            print(f"Resuming training from checkpoint {latest_checkpoint_path}")

    for fold, test_csv in enumerate(csv_files):
        print(f"Processing fold {fold + 1}/{len(csv_files)} \n")
        print(f"Using '{test_csv}' as test set for this fold. \n")

        # run = wandb.init(
        #     project =  "2d_classsification_cpap",
        #     config = {
        #         "learning_rate": learning_rate,
        #         "num_epochs": num_epochs,
        #         "fold": fold,
        #     }
        # )

        # Test dataset
        test_dataset = WebPDataset(os.path.join(dataset_folder, test_csv), with_augmentation=False)

        # Training dataset (all CSV files except the test_csv)
        train_csvs = [csv for csv in csv_files if csv != test_csv]
        train_val_dataset = load_datasets(train_csvs, dataset_folder)
        print("total training data",len(train_val_dataset))
        # Instantiate the model
        # pre_trained_model_path = '/home/aindra/Documents/2d_model/algo/model.pt'
        model = CustomModel()
        classifier = CancerClassifier(train_val_dataset, model)

        # Use the split_data method to split train_val_subset into training and validation subsets
        train_dataset, val_dataset = classifier.split_data(train_val_dataset, train_ratio=0.8)


        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

        

        # Train the model
        # print("Entered Train")
        classifier.train(train_loader, val_loader, num_epochs, learning_rate, fold, 0, checkpoint_folder, resume)

        # Test the model
        test_accuracy, test_sensitivity, test_specificity,test_ppv, test_npv = test_model(model,test_loader, fold)
        metrics = {'Fold': fold, 'Accuracy': test_accuracy, 'Sensitivity': test_sensitivity, 'Specificity': test_specificity, 'Test PPV' : test_ppv, 'Test NPV': test_npv}
        # wandb.log({'Fold': fold, 'Accuracy': test_accuracy, 'Sensitivity': test_sensitivity, 'Specificity': test_specificity, 'Test PPV' : test_ppv, 'Test NPV': test_npv})
        metrics_all_folds.append(metrics)


    # Averaging the results
    average_metrics = pd.DataFrame(metrics_all_folds).mean()
    return average_metrics, metrics_all_folds


# Main execution
dataset_folder = os.path.join(sftp_utils.sftp_path,'media/Data/2d_annot_model_lbc/dataset/train_folds_otsu_rectfd_1')
checkpoint_folder = os.path.join(sftp_utils.sftp_path,'media/Data/2d_annot_model_lbc/dataset/checkpoint_folder_lbc_resnet18_SA_rectfd_1')
os.makedirs(checkpoint_folder,exist_ok=True )
num_epochs = 100
learning_rate =  0.001
resume_training = True  # Set to True to resume training from the last checkpoint



average_metrics, metrics_all_folds = k_fold_cross_validation(dataset_folder, num_epochs, learning_rate, checkpoint_folder,resume=resume_training)

# Print or save the average metrics
print("Average Metrics Across All Folds:", average_metrics)
metrics_all_folds_df = pd.DataFrame(metrics_all_folds)
metrics_all_folds_df.to_csv(os.path.join(checkpoint_folder, 'metrics_all_folds_shuffel.csv'), index=False)