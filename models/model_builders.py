import torch
import torchvision
from torch import nn

class ViTModelConcat(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        device: torch.device,
    ):
        super().__init__()
        
		# Pretrained feature extractor
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False
        pretrained_vit.heads = nn.Linear(in_features = 768, out_features = 256).to(device)
		
		# Get automatic transforms from pretrained ViT weights
        pretrained_vit_transforms = pretrained_vit_weights.transforms()

        self.transformations = pretrained_vit_transforms
        self.feature_extractor = pretrained_vit
        self.mlp_regressor = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.GELU(),
            nn.Linear(256, 256),
		)
        self.mlp_label = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.GELU(),
            nn.Linear(256, n_targets),
		)
        self.initialize_weights()
    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.mlp_regressor[2].weight)
        nn.init.kaiming_uniform_(self.mlp_label[2].weight)
    def forward(self, inputs):
        return {
            'targets': self.mlp_label(
            torch.cat(
                (
                    self.feature_extractor(
                        self.transformations(inputs['image'].float())
					).squeeze(),
                    self.mlp_regressor(inputs['feature'].float())
                ), dim=1 
			)   
		)
		}
    


class ViTModelAdd(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        device: torch.device,
    ):
        super().__init__()
        
		# Pretrained feature extractor
        pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
        for parameter in pretrained_vit.parameters():
            parameter.requires_grad = False
        pretrained_vit.heads = nn.Linear(in_features = 768, out_features = 256).to(device)
		
		# Get automatic transforms from pretrained ViT weights
        pretrained_vit_transforms = pretrained_vit_weights.transforms()

        self.transformations = pretrained_vit_transforms
        self.feature_extractor = pretrained_vit
        self.mlp_regressor = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.GELU(),
            nn.Linear(256, 256),
		)
        self.mlp_label = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, n_targets),
		)
        self.initialize_weights()
    def initialize_weights(self):
        nn.init.kaiming_uniform_(self.mlp_regressor[2].weight)
        nn.init.kaiming_uniform_(self.mlp_label[2].weight)
    def forward(self, inputs):
        return {
             'targets': self.mlp_label(
                 self.feature_extractor(self.transformations(inputs['image'].float())) + self.mlp_regressor(inputs['feature'].float())
			 )
		}