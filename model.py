import torch
import torch.nn as nn
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, pre_trained_model_path=None):
        super(CustomModel, self).__init__()
        # Initialize ResNet18 but without the final fully connected layer
        resnet = models.resnet18(pretrained=False)  # Do not load default weights
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove the last fully connected layer and avgpool layer

        # Custom transition layer
        self.transition_layer = nn.Conv2d(512, 1024, kernel_size=1)

        # Additional layers
        self.additional_layers = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 15 * 15, 512),  # Ensure this matches the output from additional_layers
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

        if pre_trained_model_path:
            # Load the state dict
            state_dict = torch.load(pre_trained_model_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.backbone(x)
        x = self.transition_layer(x)
        x = self.additional_layers(x) 
        x = x.view(x.size(0), -1)  # Flatten the features for the classifier
        x = self.classifier(x)
        return x

# Initialization
pre_trained_model_path = '/media/dataset/annot_cases_cpap/algo/model.pt'
model = CustomModel(pre_trained_model_path)

# Load the checkpoint correctly
checkpoint_path = "/media/dataset/annot_cases_cpap/data/checkpoint_folder_cpap/best_model_fold_0.pth"
checkpoint = torch.load(checkpoint_path)


model.load_state_dict(checkpoint)  # Assuming checkpoint is directly the state dict

## Debugging: Check for mismatched keys
missing_keys = []
extra_keys = []
for key in model.state_dict().keys():
    if key not in checkpoint:
        missing_keys.append(key)
for key in checkpoint.keys():
    if key not in model.state_dict():
        extra_keys.append(key)

if missing_keys:
    print("Missing in checkpoint:", missing_keys)
if extra_keys:
    print("Extra in checkpoint:", extra_keys)
if not missing_keys and not extra_keys:
    print("All keys match perfectly.")
