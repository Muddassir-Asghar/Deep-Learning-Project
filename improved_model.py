import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = probs.gather(1, targets.view(-1, 1)).squeeze(1)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype)
                alpha_t = alpha.gather(0, targets)
            else:
                alpha_t = torch.full_like(pt, float(self.alpha))
        else:
            alpha_t = 1.0

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(2048)
        self.conv2 = nn.Conv2d(
            2048, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False
        )
        self.bn2 = nn.BatchNorm2d(2048)
        self.conv3 = nn.Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class Model(nn.Module):
    def __init__(self, vit, processor, args, num_labels=2):
        super(Model, self).__init__()
        self.vit = vit
        self.cnn = CNN()

        self.processor = processor
        self.args = args

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 1000)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(1000, num_labels)

        alpha = None
        if getattr(args, "focal_alpha", -1.0) is not None and args.focal_alpha >= 0:
            alpha = args.focal_alpha
        self.loss_fct = FocalLoss(gamma=getattr(args, "focal_gamma", 2.0), alpha=alpha)

    def forward(self, pixel_values, labels=None):
        hidden_state = self.vit(pixel_values=pixel_values).last_hidden_state
        hidden_state = hidden_state[:, 1:, :]
        batch_size, w_h, channels = hidden_state.shape
        w_h = int(math.sqrt(w_h))
        hidden_state = hidden_state.view(batch_size, w_h, w_h, channels)
        hidden_state = hidden_state.permute(0, 3, 1, 2)

        hidden_state = self.cnn(hidden_state)
        hidden_state = self.avgpool(hidden_state)
        hidden_state = torch.flatten(hidden_state, 1)

        hidden_state = self.fc(hidden_state)
        hidden_state = self.dropout(hidden_state)
        logits = self.classifier(hidden_state)

        if labels is not None:
            return self.loss_fct(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        return probs
