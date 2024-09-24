import timm
import torch.nn as nn

class ModifiedSwinTransformer(nn.Module):
    def __init__(self, num_classes_original, num_classes_large, num_classes_small):
        super(ModifiedSwinTransformer, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.model.head.fc = nn.Linear(self.model.num_features, num_classes_original)
        self.large_classifier = nn.Linear(self.model.num_features, num_classes_large)
        self.small_classifier = nn.Linear(self.model.num_features, num_classes_small)
        self.intermediate_outputs = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.intermediate_outputs[name] = output
            return hook

        self.model.layers[3].blocks[1].register_forward_hook(hook_fn('head_before_before'))
        self.model.norm.register_forward_hook(hook_fn('head_before'))

    def forward(self, x):
        y_hat = self.model(x)

        large_feature = self.intermediate_outputs['head_before_before']
        small_feature = self.intermediate_outputs['head_before']

        large_output = self.large_classifier(large_feature.mean(dim=[1,2]))  # 대분류
        small_output = self.small_classifier(small_feature.mean(dim=[1,2]))  # 중분류

        return y_hat, large_output, small_output