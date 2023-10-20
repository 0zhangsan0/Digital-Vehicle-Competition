import torch
import torch.nn as nn


class LossFunction(nn.Module):
    def __init__(self,reduction='sum'):
        super(LossFunction, self).__init__()
        self.mode=reduction

    def forward(self, pred, target):
        # 检查预测值和目标值的形状是否相同
        assert pred.size() == target.size(), "Prediction and target should have the same shape"

        batch_size=target.shape[0]
        pred=pred.reshape(-1)
        target=target.reshape(-1)
        loss = torch.tensor([item[1] - item[0] for item in zip(pred, target) if item[1] != -1],requires_grad=True)
        loss=loss.pow(2)
        loss_sum=torch.sum(loss)
        if self.mode=='sum':
            return loss_sum
        else:
            return loss_sum/batch_size


if __name__ == '__main__':
    loss_fn = LossFunction()

    # 假设我们有一些预测值 `pred` 和目标值 `target`
    pred = torch.Tensor([1, 1.5, 3, 2.5, 4, 8])  # a random tensor of shape [3, 5]
    target = torch.Tensor([1, 0, 2, 0, 0, 6])  # a random tensor of shape [3, 5]

    loss = loss_fn(pred, target)

    print(loss)
