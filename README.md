# 高光谱图像分割仓库使用教程

- 模型全在segmenations文件夹下,如果需要自己建立模型，可以参考segmenations文件夹下的模型，并记得在初始文件夹中注册，
如果forward不同，记得复写encoderdecoder类
- config文件是配置文件，可以指定encoder头和decoderhead
- datasets文件夹这样设计的好处是，兼容不同数据集
- utils是用于存放一些工具函数，损失、可视化等



metric update伪代码
```python
for each batch:
    model(x) → y_hat
    update(y_hat, y)
        → 累积 cm（混淆矩阵）
        → 收集 outputs 和 targets

on_validation_epoch_end():
    → 用 cm 计算 Pixel Acc, mIoU, Precision, Recall, F1
    → 用 outputs + targets 绘制曲线：ROC / PR / F1 vs Threshold
    → 清空状态，为下一个 epoch 做准备
```