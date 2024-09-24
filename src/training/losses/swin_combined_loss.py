import torch.nn as nn

class SwinCombinedLoss(nn.Module):
    def __init__(self, large_loss_weight=0.7, small_loss_weight=0.2, original_loss_weight=0.1):
        super(SwinCombinedLoss, self).__init__()
        self.large_loss_fn = nn.CrossEntropyLoss()
        self.small_loss_fn = nn.CrossEntropyLoss()
        self.original_loss_fn = nn.CrossEntropyLoss()
        self.large_loss_weight = large_loss_weight
        self.small_loss_weight = small_loss_weight
        self.original_loss_weight = original_loss_weight

    def forward(self, original_output, original_label, large_output, large_label, small_output, small_label):
        # 모델 출력이 튜플일 경우 첫 번째 요소만 사용
        if isinstance(original_output, tuple):
            original_output = original_output[0]
        if isinstance(large_output, tuple):
            large_output = large_output[0]
        if isinstance(small_output, tuple):
            small_output = small_output[0]
        original_loss = self.original_loss_fn(original_output, original_label)
        large_loss = self.large_loss_fn(large_output, large_label)
        small_loss = self.small_loss_fn(small_output, small_label)

        total_loss = (
            (self.large_loss_weight * large_loss) + 
            (self.small_loss_weight * small_loss) +
            (self.original_loss_weight * original_loss)
        )
        return total_loss