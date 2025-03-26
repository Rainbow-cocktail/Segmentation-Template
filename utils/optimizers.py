import torch


def get_optimizer_and_scheduler(model, train_cfgs):
    opt_name = train_cfgs.get("optimizer", "adam").lower()
    lr = train_cfgs.get("lr_rate", 1e-3)
    scheduler_type = train_cfgs.get("lr_scheduler", "step")
    scheduler_step = train_cfgs.get("lr_scheduler_step", 10)
    scheduler_gamma = train_cfgs.get("lr_scheduler_gamma", 0.1)

    # 创建优化器
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        raise ValueError(f"[get_optimizer_and_scheduler] Unsupported optimizer: {opt_name}")

    # 创建学习率调度器
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    elif scheduler_type == "multistep":
        milestones = train_cfgs.get("lr_scheduler_milestones", [30, 60])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=scheduler_gamma)
    elif scheduler_type == "cosine":
        T_max = train_cfgs.get("lr_scheduler_tmax", 50)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"[get_optimizer_and_scheduler] Unsupported scheduler type: {scheduler_type}")

    print(f"[OPTIMIZER] 使用优化器: {opt_name.upper()}, 初始学习率: {lr}")
    print(f"[SCHEDULER] 学习率调度策略: {scheduler_type.upper()}", end='')

    return {"optimizer": optimizer, "lr_scheduler": scheduler}
