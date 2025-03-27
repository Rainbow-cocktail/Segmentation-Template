from .roadSeg_datasets import HsiRoadDataset


# 注册数据集
def get_dataset(train_mode, args):
    args.update({'mode': train_mode})
    if args['name'] == 'hsi_road':
        return HsiRoadDataset(**args)
    else:
        raise NotImplementedError(f"不存在{args['name']}数据集")



