import torch


def _get_activation(activation):
    if activation is None:
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        # 二分类
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax":
        # 多分类
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d")
    return activation_fn


def iou(pr, gt, eps=1e-7, activation=None):
    activation_fn = _get_activation(activation)
    pr = activation_fn(pr)
    if activation == 'sigmoid':
        pr = (pr > 0.5).float()
        intersection = torch.sum(gt * pr)
        union = torch.sum(gt) + torch.sum(pr) - intersection + eps
        result = (intersection + eps) / union
    else:
        class_num = pr.size()[1]
        pr = torch.argmax(pr, dim=1).view(-1) + 1
        if gt.dim() == 4:
            gt = torch.argmax(gt, dim=1).view(-1) + 1
        else:
            # 如果 gt 形状是 [N, H, W]
            gt = gt.view(-1) + 1
        intersection = pr * (gt == pr).long()  # pr*mask
        area_intersection = torch.histc(intersection, bins=class_num, min=1, max=class_num)
        area_pred = torch.histc(pr, bins=class_num, min=1, max=class_num)
        area_gt = torch.histc(gt, bins=class_num, min=1, max=class_num)
        area_union = area_pred + area_gt - area_intersection
        result = area_intersection.float() / (area_union.float() + eps)
        return result

    return result


class IoUMetric(object):
    __name__ = 'iou'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        self.eps = eps
        self.activation = activation

    def __call__(self, y_pr, y_gt):
        return iou(y_pr, y_gt, eps=self.eps, activation=self.activation)


if __name__ == '__main__':

    # 假设有 3 个类别，大小为 1x3x2x2 (N, C, H, W)
    y_pred = torch.tensor([[
        [[0.1, 0.6], [0.8, 0.3]],  # 类别 0
        [[0.7, 0.2], [0.1, 0.5]],  # 类别 1
        [[0.2, 0.2], [0.1, 0.2]],  # 类别 2
    ]])  # 未经过 softmax

    # 真实标签是 2x2
    y_gt = torch.tensor([[
        [1, 0],
        [0, 1]
    ]])  # 每个像素的分类索引

    iou_metric = IoUMetric(activation='softmax')
    iou_scores = iou_metric(y_pred, y_gt)

    print("Per-class IoU:", iou_scores.numpy())  # 输出每个类别的 IoU