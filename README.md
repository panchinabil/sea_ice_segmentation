
# Sea-ice segmentation
> This repository contains code for the paper titled 'Supplementing remote sensing of ice: Deep learning-based image segmentation system for automatic detection and localization of sea-ice formations from close-range optical images'. 


# Dataset

**Folder structure**


```
.
└── data
    └── images
    └── labels
    └── validation.txt
```

```images```: Contains all the images (.jpg).

```labels```: Contains all the labels (.png).

```validation.txt```: Contains the names of the images to be used for validation (one line contains one image name, with extension).

```python
path = Path("./data")
path_img = path / "images"
path_lbl = path / "labels"
validation_file = "../validation.txt"  # Relative to path_img


img_size = 512  # 512x512 pixels
batch_size = 2
classes = [
    "Brash ice",
    "Deformed ice",
    "Floeberg",
    "Floebit",
    "Ice floe",
    "Iceberg",
    "Level ice",
    "Melt pond",
    "Open water",
    "Pancake ice",
    "Shore",
    "Sky",
    "Underwater ice",
]
```

```python
def get_label_from_img(img_file, path_lbl=path_lbl):
    return path_lbl / f"{img_file.stem}_mask.png"


def get_data(
    path_img=path_img,
    path_lbl=path_lbl,
    validation_file=validation_file,
    img_size=img_size,
    batch_size=batch_size,
    classes=classes,
):
    """Get the dataset object"""

    tfms = get_transforms(do_flip=True, max_rotate=5, max_lighting=0.1)

    data = (
        SegmentationItemList.from_folder(path_img)
        .split_by_fname_file(validation_file)
        .label_from_func(get_label_from_img, classes=classes)
        .transform(tfms, size=img_size, tfm_y=True)
        .databunch(bs=batch_size)
        .normalize(imagenet_stats)
    )

    return data
```

```python
data = get_data(
    path_img=path_img,
    path_lbl=path_lbl,
    validation_file=validation_file,
    img_size=img_size,
    batch_size=batch_size,
    classes=classes,
)
```

# Metrics

```python
def F1_score(y_pred, y_true, argmax=True, average="macro"):
    """
    A wrapper around the sklearn method `fbeta_score`.
    Computes the F-beta score between `y_pred` and `y_true`.
    """

    if argmax:
        y_pred = y_pred.argmax(dim=1)

    n = y_pred.shape[0]
    y_true = y_true.float().view(n, -1)
    y_pred = y_pred.float().view(n, -1)

    scores = torch.zeros(n)
    for i in range(n):
        scores[i] = sklearn.metrics.fbeta_score(
            to_np(y_true[i]), to_np(y_pred[i]), beta=1, average=average
        )

    return scores.mean()


def Accuracy(y_pred, y_true, argmax=True):
    """Computes accuracy between `y_pred` and `y_true`."""

    if argmax:
        y_pred = y_pred.argmax(dim=1)

    y_pred = y_pred.squeeze(1).float()
    y_true = y_true.squeeze(1).float()

    return (y_pred == y_true).float().mean()


def one_hot(y_pred, y_true, argmax=True):
    """Helper function for calcuation of IOU."""

    n, c, h, w = y_pred.shape

    range_tensor_ = to_device(
        torch.stack([torch.arange(c)] * w * h, dim=1).view(c, -1), y_pred.device
    )
    range_tensor_batch_ = to_device(
        torch.stack([range_tensor_] * n, dim=1).float(), y_pred.device
    )

    if argmax:
        y_pred = y_pred.argmax(dim=1)

    y_pred_ = to_device(torch.stack([y_pred] * c).float().view(c, n, -1), y_pred.device)
    y_true_ = to_device(
        torch.stack([y_true.squeeze(1)] * c).float().view(c, n, -1), y_pred.device
    )

    y_pred_ = (y_pred_ == range_tensor_batch_).float()
    y_true_ = (y_true_ == range_tensor_batch_).float()
    return y_pred_, y_true_, n, c, h, w


def Mean_IoU(y_pred, y_true, argmax=True, eps=1e-15):
    """Calculates mean IoU between `y_pred` and `y_true`"""

    y_pred, y_true, n, c, h, w = one_hot(y_pred, y_true, argmax)

    intersection = (y_pred * y_true).sum(dim=2).float()
    union = (y_pred + y_true).sum(dim=2).float()
    ious = (intersection + eps) / (union - intersection + eps)

    res = ious.sum(dim=1) / n
    res = res.sum() / (c)
    return tensor(res)
```

```python
metrics = [Mean_IoU, Accuracy, F1_score]
```

# Neural network

```python
pspnet = PSPNet(num_classes=len(classes), backbone="resnet152", pretrained=False)
```

# Postprocessing 

```python
postproc_fn = conv_crf
```

# Training

```python
init = True  # Initialize the neural network using Kaiming normalization
wd = 1e-2  # Weight decay parameter
half_prec = True  # Half precision training
epochs = [20, 60]  # Stage 1: 20 training epochs, Stage 2: 60 training epochs
lr1 = 5e-3  # Learning rate for stage 1
lr2 = [lr / 40, lr / 4]  # Learning rates for stage 2
pct_starts = [
    0.9,
    0.9,
]  # Increase the lr from a min value to the given lr in first 90% of the training iterations in both stage 1 and 2
```

```python
def get_learner(data, model, init, metrics, wd):
    model = to_device(model, data.device)
    learn = Learner(data=data, model=model, metrics=metrics, wd=wd)
    learn.split(model.split_model)

    if model.pretrained and len(learn.layer_groups) > 1:
        learn.freeze()
    if init:
        apply_init(model, torch.nn.init.kaiming_normal_)

    return learn


def train(data, model, init, metrics, wd, half_prec, epochs, lr1, lr2, pct_starts, postproc_fn):
      
    model_name = type(model).__name__
    learn = get_learner(data, model, init, metrics, wd)
    if half_prec: learn = learn.to_fp16()
    
    #stage 1
    print(f"Stage 1 training...")
    name1 = f'Stage_1-{model_name}"
    learn.fit_one_cycle(epochs[0], slice(lr1), pct_start=pct_starts[0], 
                        callbacks=[SaveModelCallback(learn, every='improvement', monitor='meanIOU-ConvCRF', mode='max', name=name1), 
                        CSVLogger(learn=learn, filename=f"../../results/{model_name}"),  #filename is relative to the path_img
                        *get_metrics_callbacks(learn, metrics, postproc_fn)])

    #stage 2
    learn.unfreeze()
    print(f"Stage 2 training...")
    name2 = f'Stage_2-{model_name}"
    learn.fit_one_cycle(epochs[1], slice(lr2), pct_start=pct_starts[1], 
                        callbacks=[SaveModelCallback(learn, every='improvement', monitor='meanIOU-ConvCRF', mode='max', name=name2), 
                        CSVLogger(learn=learn, filename=f"../../results/{model_name}", append=True),  #filename is relative to the path_img
                        *get_metrics_callbacks(learn, metrics, postproc_fn)])
    
    learn.load(name2);
                                   
    learn.name1, learn.name2 = name1, name2
    
    return learn
```

```python
learn = train(
    data, model, init, metrics, wd, half_prec, epochs, lr1, lr2, pct_starts, postproc_fn
)
```

# References

- The implementation of the deep learning models was derived from https://github.com/yassouali/pytorch-segmentation

- The implementation of the convolutional conditional random field used for postprocessing was derived from https://github.com/MarvinTeichmann/ConvCRF 
