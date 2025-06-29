# model/fetalnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- This is your existing FetalNet model code (Unchanged) ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class FetalNet(nn.Module):
    def __init__(self, num_classes_model=6):
        super(FetalNet, self).__init__()
        self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        for param in self.base.parameters():
            param.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(256),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # This is our target layer for activations
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes_model)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.conv(x)
        x = self.classifier(x)
        return x

# --- NEW, UNIQUE CODE FOR XAI ---
# This class wraps the original model to expose the activations we need.
# It's a clean way to get intermediate data without changing the original model.
class FetalNetXAI(nn.Module):
    def __init__(self, model):
        super(FetalNetXAI, self).__init__()
        self.model = model
        # We need the activations from the final conv layer before the classifier
        self.feature_extractor = self.model.conv

    def forward(self, x):
        # Get the feature maps from our target layer
        features = self.feature_extractor(self.model.base(x))
        # Get the final output
        output = self.model.classifier(features)
        return output, features



























# # model/fetalnet.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models

# class SEBlock(nn.Module):
#     """Squeeze-and-Excitation Block."""
#     def __init__(self, channels, reduction=8):
#         super(SEBlock, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(channels, channels // reduction)
#         self.fc2 = nn.Linear(channels // reduction, channels)

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.pool(x).view(b, c)
#         y = F.relu(self.fc1(y))
#         y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
#         return x * y

# class FetalNet(nn.Module):
#     """The main, complete FetalNet model architecture."""
#     def __init__(self, num_classes_model=6):
#         super(FetalNet, self).__init__()
#         self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
#         for param in self.base.parameters():
#             param.requires_grad = False

#         self.conv = nn.Sequential(
#             nn.Conv2d(1280, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             SEBlock(256),
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes_model)
#         )

#     def forward(self, x):
#         x = self.base(x)
#         x = self.conv(x)
#         x = self.classifier(x)
#         return x

# class FetalNet_No_SE(nn.Module):
#     """Ablation model: FetalNet without the SEBlock."""
#     def __init__(self, num_classes_model=6):
#         super(FetalNet_No_SE, self).__init__()
#         self.base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
#         for param in self.base.parameters():
#             param.requires_grad = False

#         self.conv = nn.Sequential(
#             nn.Conv2d(1280, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             # SEBlock(256) is removed for this ablation study.
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1)
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes_model)
#         )

#     def forward(self, x):
#         x = self.base(x)
#         x = self.conv(x)
#         x = self.classifier(x)
#         return x