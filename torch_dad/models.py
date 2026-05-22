import torch
import torch.nn as nn
from .layers import DADLinear

class DADModel(nn.Module):
    """
    Base DAD Neural Network Container.
    Inherits from nn.Module, automatically scanning and managing any nested DADLinear layers.
    """
    def __init__(self):
        super().__init__()
        
    @property
    def dad_layers(self):
        """Dynamically scans and returns all DADLinear layers in the model in registration order."""
        return [module for module in self.modules() if isinstance(module, DADLinear)]

    def forward_inference(self, x):
        """
        Runs standard inference through the model without target propagation overhead.
        Expects a final linear layer named `self.classifier` in subclasses.
        """
        x = x.view(x.size(0), -1)
        with torch.no_grad():
            for layer in self.dad_layers:
                out, _ = layer(x)
                x = out
                
        # Subclasses must define self.classifier (e.g. standard nn.Linear final head)
        if hasattr(self, 'classifier'):
            return self.classifier(x)
        else:
            raise AttributeError("DADModel subclasses must define self.classifier as their final output head.")
