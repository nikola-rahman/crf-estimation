"""PyTorch DNN model for CRF prediction."""

import torch
import torch.nn as nn

import config


class CRFPredictor(nn.Module):
    """5 hidden-layer DNN for CRF class prediction.

    Architecture from paper: input → 100 → 68 → 50 → 100 → 50 → num_classes
    All hidden layers use ReLU activation.
    Output is raw logits (use CrossEntropyLoss which includes softmax).
    """

    def __init__(self, input_dim: int = config.NUM_INPUT_FEATURES,
                 num_classes: int = config.NUM_CRF_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 68),
            nn.ReLU(),
            nn.Linear(68, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
