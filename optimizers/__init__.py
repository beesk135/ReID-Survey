import torch 

def make_optimizer(model,opt, lr,weight_decay,momentum=0.9,nesterov=True):
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        optimizer = getattr(torch.optim,'Adam')(model.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
    else:
        optimizer = getattr(torch.optim, opt)(model.parameters(),lr=lr,weight_decay=weight_decay)
    return optimizer

def make_optimizer_partial(weights,opt, lr,weight_decay,momentum=0.9,nesterov=True):
    if opt == 'SGD':
        optimizer = getattr(torch.optim, opt)(weights,lr=lr,weight_decay=weight_decay, momentum=momentum,nesterov=nesterov)
    elif opt == 'AMSGRAD':
        optimizer = getattr(torch.optim,'Adam')(weights,lr=lr,weight_decay=weight_decay,amsgrad=True)
    else:
        optimizer = getattr(torch.optim, opt)(weights,lr=lr,weight_decay=weight_decay)
    return optimizer