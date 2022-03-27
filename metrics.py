import torch

#Define mean IoU metric
#Taken from fastai v1

def one_hot(input, targs, classes=None, argmax=True):
    n,c,h,w = input.shape
    if classes==None: classes=c
    range_tensor_ = torch.stack([torch.arange(classes)]*w*h, dim=1).view(classes,-1).to(input.device, non_blocking=True)
    range_tensor_batch_ = torch.stack([range_tensor_]*n, dim = 1).float().to(input.device, non_blocking=True)
    
    if argmax: input = input.argmax(dim=1)
    
    input_ = torch.stack([input]*classes).float().view(classes,n, -1).to(input.device, non_blocking=True)
    targs_ = torch.stack([targs.squeeze(1)]*classes).float().view(classes,n, -1).to(input.device, non_blocking=True)

    input_ = (input_ == range_tensor_batch_).float()
    targs_ = (targs_ == range_tensor_batch_).float()
    return input_, targs_, classes, n, h, w


def IOU(input, targs, classes=None, argmax=True, eps = 1e-15):
  
    input_, targs_, classes, n, h, w = one_hot(input, targs, classes, argmax)
    intersect_ = (input_*targs_).sum(dim = 2).float()
    union_ = (input_+targs_).sum(dim = 2).float()
    ious = (intersect_ + eps)/ (union_-intersect_+eps)
    res = ious.sum(dim = 1)/n
    
    res = res.sum()/(classes)
    
    return torch.tensor(res)