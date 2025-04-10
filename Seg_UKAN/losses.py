import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'CrossEntropyLoss', 'LovaszHingeLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
'''
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Cross-entropy loss for multiclass segmentation (using one-hot encoded target)
        target_idx = target.argmax(dim=1)  # Convert one-hot target to class indices
        ce_loss = F.cross_entropy(input, target_idx)  # input: (batch_size, num_classes, H, W), target: (batch_size, H, W)

        smooth = 1e-5
        num_classes = input.size(1)  # Number of classes
        num = target.size(0)

        # Reshape input to [batch_size, num_classes, -1] (flatten spatial dimensions)
        input = input.view(num, num_classes, -1)
        target = target.view(num, num_classes, -1)  # Target is already one-hot encoded

        # Compute dice loss for each class
        dice_loss = 0.0
        for i in range(num_classes):
            input_class = input[:, i]
            target_class = target[:, i]  # Target for class i (already one-hot encoded)

            intersection = (input_class * target_class).sum(dim=1)
            dice = (2. * intersection + smooth) / (input_class.sum(dim=1) + target_class.sum(dim=1) + smooth)
            dice_loss += (1 - dice).mean()  # Average dice loss across batch

        # Combine cross-entropy and dice loss
        return ce_loss + dice_loss
'''
'''
class CrossEntropyDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Cross-entropy loss for multiclass segmentation (using one-hot encoded target)
        target_idx = target.argmax(dim=1)  # Convert one-hot target to class indices
        ce_loss = F.cross_entropy(input, target_idx)  # input: (batch_size, num_classes, H, W), target: (batch_size, H, W)

        smooth = 1e-5
        num_classes = input.size(1)  # Number of classes
        batch_size = target.size(0)

        # Reshape input to [batch_size, num_classes, -1] (flatten spatial dimensions)
        input = input.view(batch_size, num_classes, -1)
        target = target.view(batch_size, num_classes, -1)  # Target is already one-hot encoded

        # Compute dice loss for each class
        dice_loss = 0.0
        for i in range(num_classes):
            input_class = input[:, i]
            target_class = target[:, i]  # Target for class i (already one-hot encoded)

            intersection = (input_class * target_class)
            dice = (2. * intersection.sum(dim=1) + smooth) / (input_class.sum(dim=1) + target_class.sum(dim=1) + smooth)
            dice_loss += (1 - dice).mean()  # Average dice loss across batch

        # Combine cross-entropy and dice loss
        return ce_loss + dice_loss
'''
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Cross-entropy loss for multiclass segmentation (using one-hot encoded target)
        target_idx = target.argmax(dim=1)  # Convert one-hot target to class indices
        ce_loss = F.cross_entropy(input, target_idx)  # input: (batch_size, num_classes, H, W), target: (batch_size, H, W)

        return ce_loss
        
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
