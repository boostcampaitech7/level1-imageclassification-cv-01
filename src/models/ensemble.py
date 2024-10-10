import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
class EnsembleModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        # Load State_dict
        ckpt1 = torch.load("result/epoch=50-step=9282.ckpt")
        # ckpt2 = torch.load('../result/swin_base_patch4_window7_224/epoch=29-step=19170.ckpt')
        state_dict1 = {'.'.join(key.split('.')[2:]): val for key, val in ckpt1['state_dict'].items()}
        # state_dict2 = {'.'.join(key.split('.')[2:]): val for key, val in ckpt2['state_dict'].items()}
        # Make Branch
        self.clip_base = timm.create_model('vit_giant_patch14_clip_224', pretrained=True, num_classes=num_classes)
        self.swin_base = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=num_classes)
        self.convnext_base = timm.create_model('convnextv2_huge', pretrained=True, num_classes=num_classes)
        # self.rex_base.load_state_dict(state_dict1)
        self.swin_base.load_state_dict(state_dict1)
        # Remove FC layer
        clip_in_features, swin_in_feuatres, convnext_in_features = self.clip_base.head.in_features, self.swin_base.head.fc.in_features, self.convnext_base.head.fc.in_features
        self.clip_base.head = nn.Identity()
        self.swin_base.head.fc = nn.Identity()
        self.convnext_base.head.fc = nn.Identity()
        # Freeze base models
        for model in [self.clip_base, self.swin_base, self.convnext_base]:
            for param in model.parameters():
                param.requires_grad = False
        # Create branches
        clip_branch_output_dim, swin_branch_output_dim, convnext_branch_output_dim = 512, 384, 768
        self.clip_branch = self.create_branch(self.clip_base, clip_in_features, clip_branch_output_dim)
        self.swin_branch = self.create_branch(self.swin_base, swin_in_feuatres, swin_branch_output_dim)
        self.convnext_branch = self.create_branch(self.convnext_base, convnext_in_features, convnext_branch_output_dim)
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(clip_branch_output_dim+swin_branch_output_dim+convnext_branch_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def create_branch(self, base_model, result_features, target_features):
        return nn.Sequential(
            base_model,
            nn.Flatten(start_dim=1),
            nn.Linear(result_features, target_features),
            nn.ReLU(),
            nn.Dropout(0.5)
    )
    def forward(self, x):
        # Extract features from each branch
        clip_features = self.clip_branch(x)
        swin_features = self.swin_branch(x)
        convnext_feautres = self.convnext_branch(x)
        # Concatenate features
        combined_features = torch.cat((clip_features, swin_features, convnext_feautres), dim=1)
        output = self.fc_layers(combined_features)
        return output