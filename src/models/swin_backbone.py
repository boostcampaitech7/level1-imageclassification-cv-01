import timm
import torch.nn as nn

class ModifiedSwinTransformer(nn.Module):
    def __init__(self, num_classes_original, num_classes_large, num_classes_small):
        super(ModifiedSwinTransformer, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        # 변경 전
        # self.model.head.fc = nn.Linear(self.model.num_features, num_classes_original)
        # self.large_classifier = nn.Linear(self.model.num_features, num_classes_large)
        # self.small_classifier = nn.Linear(self.model.num_features, num_classes_small)
        # 변경 후
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes_original)
        self.large_classifier = nn.Linear(self.model.layers[2].blocks[17].mlp.fc2.out_features, num_classes_large)
        self.small_classifier = nn.Linear(self.model.layers[3].blocks[1].mlp.fc2.out_features, num_classes_small)

        self.intermediate_outputs = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.intermediate_outputs[name] = output
            return hook
        # 변경 전
        # self.model.layers[3].blocks[1].register_forward_hook(hook_fn('head_before_before'))
        # self.model.norm.register_forward_hook(hook_fn('head_before'))
        # 변경 후
        self.model.layers[2].register_forward_hook(hook_fn('large'))
        self.model.layers[3].register_forward_hook(hook_fn('small'))

    def forward(self, x):
        y_hat = self.model(x)
        # 변경 전
        # large_feature = self.intermediate_outputs['head_before_before']
        # small_feature = self.intermediate_outputs['head_before']
        # 변경 후
        large_feature = self.intermediate_outputs['large']
        small_feature = self.intermediate_outputs['small']

        large_output = self.large_classifier(large_feature.mean(dim=[1,2]))  # 대분류
        small_output = self.small_classifier(small_feature.mean(dim=[1,2]))  # 중분류

        return y_hat, large_output, small_output