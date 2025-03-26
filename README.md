# 高光谱图像分割仓库使用教程

- 模型全在segmenations文件夹下,如果需要自己建立模型，可以参考segmenations文件夹下的模型，并记得在初始文件夹中注册，
如果forward不同，记得复写encoderdecoder类
- config文件是配置文件，可以指定encoder头和decoderhead
- datasets文件夹这样设计的好处是，兼容不同数据集
- utils是用于存放一些工具函数，损失、可视化等