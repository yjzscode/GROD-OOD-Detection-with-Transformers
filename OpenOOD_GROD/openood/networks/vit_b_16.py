import torch
from torchvision.models.vision_transformer import VisionTransformer, vit_b_16, ViT_B_16_Weights
import torchvision
# image_size, return_feature
from transformers import ViTImageProcessor, ViTForImageClassification
import torch.nn as nn

# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# model.config.classifier = 'mlp'
# model.config.num_labels = 10
# # print(model.get_output_embeddings)
# # print(model.classifier)
# model.classifier = nn.Linear(768,10)

# net = vit_b_16(weights = ("pretrained", ViT_B_16_Weights.IMAGENET1K_V1))
# net.hidden_dim = 768
# net.mlp_dim = 3072

class ViT_B_16(VisionTransformer):
    def __init__(self,                 
                 image_size=224,
                 patch_size=16,
                 num_layers=12,
                 num_heads=12,
                 hidden_dim=768,
                 mlp_dim=3072,
                 num_classes=1000):
        super(ViT_B_16, self).__init__(image_size=image_size,
                                       patch_size=patch_size,
                                       num_layers=num_layers,
                                       num_heads=num_heads,
                                       hidden_dim=hidden_dim,
                                       mlp_dim=mlp_dim,
                                       num_classes=num_classes,
                                       )
        self.feature_size = hidden_dim

    def forward(self, x, return_feature=True):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        # print(x.size())
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        # print(x.size())
        # print(self.heads(x).size())
        if return_feature:
            return self.heads(x), x
        else:
            return self.heads(x)

    def forward_threshold(self, x, threshold):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        feature = x.clip(max=threshold)
        logits_cls = self.heads(feature)

        return logits_cls

    def get_fc(self):
        fc = self.heads[0]
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.heads[0]
