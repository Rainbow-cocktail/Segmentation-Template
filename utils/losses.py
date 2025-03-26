import torch
import torch.nn.functional as F
import warnings


def to_one_hot(y, num_classes):
    """
    Convert a tensor of class indices to a one-hot encoded tensor.
    
    :param y: Tensor of class indices. shape: (B, H, W)
    :param num_classes: Number of classes.
    :return: One-hot encoded tensor.
    """
    return F.one_hot(y, num_classes).permute(0, 3, 1, 2).float()


class FocalLoss(torch.nn.Module):
    """二分类 + one_hot label 的 Focal Loss"""
    __name__ = 'focal_loss'

    def __init__(self, alpha=0.5, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 控制正负样本权重
        self.gamma = gamma  # 控制难样本的放大因子
        self.reduction = reduction  # 是否求平均

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # 检查是否是二分类
        if y_pred.shape[1] > 2:
            raise ValueError(
                f"[FocalLoss] 当前版本只支持二分类任务（C=1 或 2），"
                f"检测到通道数 C = {y_pred.shape[1]}，请使用 focal_multi 替代。"
            )

        # 处理 y_target 非 one-hot 的情况
        if y_target.shape != y_pred:
            warnings.warn(f"[FocalLoss] 你的标签 y_target 应为 one-hot 编码，"
                          f"当前形状: {y_target.shape}，但模型输出形状为: {y_pred.shape}。\n"
                          f"如果你是多分类任务，请使用 'focal_multi' 替代 'focal'。")
            num_classes = y_pred.shape[1]
            y_target = F.one_hot(y_target, num_classes).permute(0, 3, 1, 2).float()

        one = torch.ones_like(y_pred, dtype=y_pred.dtype)
        y_pred = torch.clamp(y_pred, 1e-8, 1. - 1e-8)
        loss0 = -self.alpha * (one - y_pred) ** self.gamma * y_target * torch.log(y_pred)
        loss1 = -(1. - self.alpha) * y_pred ** self.gamma * (one - y_target) * torch.log(one - y_pred)
        if self.reduction == 'mean':
            loss_ = torch.mean(loss0 + loss1)
        elif self.reduction == 'sum':
            loss_ = torch.sum(loss0 + loss1)
        else:
            loss_ = loss0 + loss1
        return loss_


class FocalLossMultiClass(torch.nn.Module):
    """多分类 Focal Loss"""
    __name__ = 'focal_multi'

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        y_pred: [B, C, H, W] (logits)
        y_target: [B, H, W] (int labels)
        """
        logpt = F.log_softmax(y_pred, dim=1)  # [B, C, H, W]
        pt = torch.exp(logpt)

        target = y_target.unsqueeze(1)  # [B, 1, H, W]
        logpt = logpt.gather(1, target).squeeze(1)  # [B, H, W]
        pt = pt.gather(1, target).squeeze(1)  # [B, H, W]

        loss = -self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CrossEntropy2D(torch.nn.Module):
    __name__ = 'ce'

    def __init__(self, weight=None, reduction='mean'):
        """
        PyTorch 的交叉熵损失函数，支持 2D 图像分割任务
        :param weight: 各类别的权重
        :param reduction: 损失值的计算方式, 'mean' 或 'sum'
        """

        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_logits: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """ logsoftmax+nll
        :param y_logits: [N, C, H, W] 预测的 logits
        :param y_target: [N, H, W] 真实类别索引（整数）
        :return: cross_entropy loss
        """

        return F.cross_entropy(y_logits, y_target, reduction=self.reduction, weight=self.weight)


class BinaryCrossEntropy(torch.nn.Module):
    __name__ = 'bec'

    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, y_logits: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        sigmoid + binary_cross_entropy
        y_logits: [B, 1, H, W] or [B, H, W]  → raw logits
        y_target: [B, H, W] or [B, 1, H, W]  → int labels (0 or 1)
        """
        if y_target.shape != y_logits.shape:
            if y_logits.shape[1] != 1:
                raise ValueError(
                    f"[BinaryCrossEntropy] 输出通道应为 1（得到 C={y_logits.shape[1]}），"
                    "确保是二分类 sigmoid 输出！"
                )
            y_target = y_target.unsqueeze(1).float()  # [B, 1, H, W]
        loss = F.binary_cross_entropy_with_logits(y_logits, y_target, reduction=self.reduction, weight=self.weight)
        return loss


class DiceLoss(torch.nn.Module):
    """
    Note:
        - y_pred should be raw logits or softmax probs: [B, C, H, W]
        - y_target should be class indices: [B, H, W]
        - Supports both binary and multi-class cases
    """
    __name__ = 'dice'

    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        """
        :param y_pred: B, num_classes, H, W
        :param y_true: B, H, W
        :return:
        """
        nums_classes = y_pred.size(1)
        y_target_onehot = to_one_hot(y_target, nums_classes).to(y_pred.device)

        y_pred = torch.softmax(y_pred, dim=1)  # 对每个 pixel 上的每个类别进行 softmax
        dims = (0, 2, 3)
        intersection = torch.sum(y_pred * y_target_onehot, dims)  # [C]
        union = torch.sum(y_pred, dims) + torch.sum(y_target_onehot, dims)  # [C]
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dice.mean()


class ComboLoss(torch.nn.Module):
    __name__ = 'combo'

    def __init__(self, alpha=0.5, weight=None, reduction='mean', smooth=1e-8):
        super().__init__()
        self.alpha = alpha
        self.ce = CrossEntropy2D(weight=weight, reduction=reduction)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(y_pred, y_target) + (1 - self.alpha) * self.dice(y_pred, y_target)


def get_loss(loss_name, loss_args):
    if loss_name == 'ce':
        return CrossEntropy2D(
            weight=loss_args.get('weight', None),
            reduction=loss_args.get('reduction', 'mean')
        )
    elif loss_name == 'bce':
        return BinaryCrossEntropy(
            weight=loss_args.get('weight', None),
            reduction=loss_args.get('reduction', 'mean')
        )
    elif loss_name == 'focal':
        return FocalLoss(
            alpha=loss_args.get('alpha', 0.5),
            gamma=loss_args.get('gamma', 2.0),
            reduction=loss_args.get('reduction', 'mean')
        )
    elif loss_name == 'focal_multi':
        return FocalLossMultiClass(
            alpha=loss_args.get('alpha', 1.0),
            gamma=loss_args.get('gamma', 2.0),
            reduction=loss_args.get('reduction', 'mean')
        )
    elif loss_name == 'dice':
        return DiceLoss(
            smooth=loss_args.get('smooth', 1.0)
        )
    elif loss_name == 'combo':
        return ComboLoss(
            alpha=loss_args.get('alpha', 0.5)
        )
    else:
        raise ValueError(f"Unsupported loss_name: {loss_name}")



if __name__ == "__main__":
    import torch

    # 模拟高光谱输入：B=4, C=25 → 输出 C=10 类
    B, C_in, C_out, H, W = 4, 25, 10, 160, 160
    logits = torch.randn([B, C_out, H, W], dtype=torch.float32, requires_grad=True).cuda()
    target = torch.randint(0, C_out, (B, H, W), dtype=torch.int64).cuda()

    loss_fn = get_loss('focal_multi', {'alpha': 1.0, 'gamma': 2.0, 'reduction': 'mean'})
    loss = loss_fn(logits, target)

    print("✅ FocalLossMultiClass 测试通过")
    print("Loss 值:", loss.item())

    loss.backward()  # 也可以反向传播测试梯度
    print("✅ backward 成功")

    # ✅ Dice Loss 测试（需内部 one-hot）
    dice_loss_fn = get_loss('dice', {'smooth': 1.0})
    loss_dice = dice_loss_fn(logits, target)
    print("✅ DiceLoss 测试通过")
    print("Loss 值:", loss_dice.item())
    loss_dice.backward()
    print("✅ Dice backward 成功")

    # ✅ Combo Loss 测试（CE + Dice）
    combo_loss_fn = get_loss('combo', {'alpha': 0.5})
    loss_combo = combo_loss_fn(logits, target)
    print("✅ ComboLoss 测试通过")
    print("Loss 值:", loss_combo.item())
    loss_combo.backward()
    print("✅ Combo backward 成功")
