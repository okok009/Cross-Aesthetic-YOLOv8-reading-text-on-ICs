import torch

def sgd(params, lr_rate, momentum, milestones:list = None, step = 0.1, warmup_milestones:list = None, war_step = 2):
    
    optimizer = torch.optim.SGD(params, lr=lr_rate, momentum=momentum)

    if milestones is not None and warmup_milestones is None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, step)
        return optimizer, lr_scheduler
    
    elif warmup_milestones is not None and milestones is None:
        warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer, warmup_milestones, war_step)
        return optimizer, warmup

    elif warmup_milestones is not None and milestones is not None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)
        warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer, warmup_milestones, 2)
        return optimizer, lr_scheduler, warmup
    
    else:
        return optimizer
    
def adam(params, lr_rate, betas:tuple, milestones:list = None, step = 0.1, warmup_milestones:list = None, warm_step = 2):
    
    optimizer = torch.optim.Adam(params, lr_rate, betas)

    if milestones is not None and warmup_milestones is None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, step)
        return optimizer, lr_scheduler
    
    elif warmup_milestones is not None and milestones is None:
        warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer, warmup_milestones, warm_step)
        return optimizer, warmup

    elif warmup_milestones is not None and milestones is not None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)
        warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer, warmup_milestones, 2)
        return optimizer, lr_scheduler, warmup
    
    else:
        return optimizer