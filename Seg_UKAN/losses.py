import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'CrossEntropyLoss', 'LovaszHingeLoss', 'FocalLoss']


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

    def forward(self, input, target, **kwargs):
        # Cross-entropy loss for multiclass segmentation (using one-hot encoded target)
        target_idx = target.argmax(dim=1)  # Convert one-hot target to class indices
        ce_loss = F.cross_entropy(input, target_idx)  # input: (batch_size, num_classes, H, W), target: (batch_size, H, W)

        return ce_loss
    
    
def substitute_values(x: torch.Tensor, values, unique=None):
    """
    Substitute values in a tensor with the given values
    :param x: the tensor
    :param unique: the unique values to substitute
    :param values: the values to substitute with
    :return: the tensor with the values substituted
    """
    if unique is None:
        unique = x.unique()
    lt = torch.full((unique.max() + 1,), -1, dtype=values.dtype, device=x.device)
    lt[unique] = values
    return lt[x]
    
    
def get_weight_matrix_from_labels(labels, num_classes, ignore_index=-100):  
    there_is_ignore = ignore_index in labels
    if there_is_ignore:
        weight_labels = labels.clone()
        weight_labels += 1
        weight_labels[weight_labels == ignore_index + 1] = 0
        weight_num_classes = num_classes + 1
    else:
        weight_labels = labels
        weight_num_classes = num_classes
    weights = torch.ones(weight_num_classes, device=labels.device)
    classes, counts = weight_labels.unique(return_counts=True)
    classes = classes.long()
    if there_is_ignore:
        weights[classes] = 1 / torch.log(1.1 + counts / counts.sum())
        weights[0] = 0
        class_weights = weights[1:]
    else:
        weights[classes] = 1 / torch.log(1.1 + counts / counts.sum())
        class_weights = weights
    wtarget = substitute_values(
        weight_labels,
        weights,
        unique=torch.arange(weight_num_classes, device=labels.device),
    )
    return wtarget, class_weights
    
    
class FocalLoss(nn.Module):
    def __init__(
        self, gamma: float = 2.0, class_weighting=False, **kwargs
    ):
        super().__init__()
        self.gamma = gamma
        self.class_weighting = class_weighting

    def __call__(self, x, target, weight_matrix=None, **kwargs):
        if self.class_weighting:
            num_classes = x.shape[1]
            weight_matrix, _ = get_weight_matrix_from_labels(
                target, num_classes
            )
        
        ce_loss = F.cross_entropy(x, target, reduction="none")
        pt = torch.exp(-ce_loss)
        if weight_matrix is not None:
            focal_loss = torch.pow((1 - pt), self.gamma) * weight_matrix * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return torch.mean(focal_loss)
    
        
class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
