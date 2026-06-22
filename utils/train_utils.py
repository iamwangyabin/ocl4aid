import timm
from torch import optim

from models import MODELS


def select_optimizer(opt_name, lr, model):
    opt_name = (opt_name or "").lower()

    if opt_name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif opt_name == "adamw":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif opt_name == 'adam_adapt':
        fc_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc.' in name:  # If the parameter is from a fully-connected layer
                    fc_params.append(param)
                else:  # All other layers
                    other_params.append(param)
        opt = optim.Adam([
                        {'params': fc_params, 'lr': lr},       # Learning rate lr1 for fully-connected layers
                        {'params': other_params, 'lr': lr*5}     # Learning rate lr2 for all other layers
                    ], weight_decay=0)
    elif opt_name == "sgd":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4
        )
    elif opt_name == 'sgd_sl':
        fc_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'fc.' in name:  # If the parameter is from a fully-connected layer
                    fc_params.append(param)
                else:  # All other layers
                    other_params.append(param)
        opt = optim.SGD([
                        {'params': other_params, 'lr': lr},       # Learning rate lr1 for fully-connected layers
                        {'params': fc_params, 'lr': 0.005}     # Learning rate lr2 for all other layers
                    ], weight_decay=5e-4)
    else:
        raise NotImplementedError("Please select the opt_name [adam, adamw, adam_adapt, sgd, sgd_sl]")
    return opt

def select_scheduler(sched_name, opt, hparam=None):
    sched_name = (sched_name or "").lower()
    if isinstance(hparam, dict):
        gamma = hparam.get("gamma", 0.9999)
        t_max = max(1, int(hparam.get("t_max", 1) or 1))
        eta_min = float(hparam.get("eta_min", 0.0) or 0.0)
    else:
        gamma = hparam
        t_max = 1
        eta_min = 0.0

    if sched_name in {"cosine", "cosine_annealing", "cosineannealing"}:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
    elif "exp" in sched_name:
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    elif sched_name in {"cos", "cos_warm_restarts"}:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2)
    elif sched_name == "anneal":
        scheduler = optim.lr_scheduler.ExponentialLR(opt, 1 / 1.1, last_epoch=-1)
    elif sched_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 60, 80, 90], gamma=0.1)
    elif sched_name == "const":
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    elif sched_name == "sam":
        scheduler = optim.lr_scheduler.LambdaLR(opt.base_optimizer, lambda iter: 1)
    elif sched_name == "fam":
        scheduler = optim.lr_scheduler.LambdaLR(opt.base_optimizer, lambda iter: 1)
    else:
        scheduler = optim.lr_scheduler.LambdaLR(opt, lambda iter: 1)
    return scheduler

def select_model(method, backbone, num_classes=None, n_tasks=None, kwargs=None):
    import logging
    logger = logging.getLogger()
    kwargs = dict(kwargs or {})
    pretrained = bool(kwargs.pop("pretrained", True))

    if method=="slca":
        import models.vit as vit
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone):
            logger.info(f'Using custom ViT model: {backbone}')
            model = getattr(vit, backbone)(
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=0.,
                drop_path_rate=0.,
            )
        else:
            logger.info(f'Using timm model: {backbone}')
            model = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=0.,
                drop_path_rate=0.,
                drop_block_rate=None
            )
    elif method in MODELS.keys():
        model = MODELS[method](
            backbone_name=backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            task_num=n_tasks,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Unsupported method: {method}")

    return model
