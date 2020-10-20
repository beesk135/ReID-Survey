import torch

def make_optimizers(cfg, model, criterion):
    optimizer = {}
    params = []
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'ADAM':
        optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, amsgrad=cfg.SOLVER.USE_AMSGRAD)
    else:
        optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    
    if cfg.SOLVER.CENTER_LOSS.USE:
        optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LOSS.LR, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer
#
# def make_optimizer(model,opt, lr,weight_decay,momentum=0.9,nesterov=True):
#     if opt == 'SGD':
#         optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
#     elif opt == 'AMSGRAD':
#         optimizer = getattr(torch.optim,'Adam')(model.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
#     else:
#         optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay)
#     return optimizer
#
# def make_optimizer_partial(weights,opt, lr,weight_decay,momentum=0.9,nesterov=True):
#     if opt == 'SGD':
#         optimizer = getattr(torch.optim, opt)(weights,lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
#     elif opt == 'AMSGRAD':
#         optimizer = getattr(torch.optim,'Adam')(weights,lr=lr,weight_decay=weight_decay,amsgrad=True)
#     else:
#         optimizer = getattr(torch.optim, opt)(weights,lr=lr,weight_decay=weight_decay)
#     return optimizer
