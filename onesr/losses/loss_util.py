import functools
import oneflow as flow
from oneflow.nn import functional as F

_reduction_modes = ["none", "mean", "sum"]


def get_enum(reduction):
    if reduction == "none":
        ret = 0
    elif reduction == "mean":
        ret = 1
    elif reduction == "elementwise_mean":
        warnings.warn(
            "reduction='elementwise_mean' is deprecated, please use reduction='mean' instead."
        )
        ret = 1
    elif reduction == "sum":
        ret = 2
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    return ret


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = get_enum(reduction)
    # none: 0, mean / elementwise_mean: 1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean"):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of Pyflow. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == "sum":
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == "mean":
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import oneflow as flow
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = flow.Tensor([0, 2, 3])
    >>> target = flow.Tensor([1, 1, 1])
    >>> weight = flow.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


def get_local_weights(residual, ksize):
    """Get local weights for generating the artifact map of LDL.

    It is only called by the `get_refined_artifact_map` function.

    Args:
        residual (Tensor): Residual between predicted and ground truth images.
        ksize (Int): size of the local window.

    Returns:
        Tensor: weight for each pixel to be discriminated as an artifact pixel
    """

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = (
        flow.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
        .squeeze(-1)
        .squeeze(-1)
    )

    return pixel_level_weight


def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):
    """Calculate the artifact map of LDL
    (Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution. In CVPR 2022)

    Args:
        img_gt (Tensor): ground truth images.
        img_output (Tensor): output images given by the optimizing model.
        img_ema (Tensor): output images given by the ema model.
        ksize (Int): size of the local window.

    Returns:
        overall_weight: weight for each pixel to be discriminated as an artifact pixel
        (calculated based on both local and global observations).
    """

    residual_ema = flow.sum(flow.abs(img_gt - img_ema), 1, keepdim=True)
    residual_sr = flow.sum(flow.abs(img_gt - img_output), 1, keepdim=True)

    patch_level_weight = flow.var(
        residual_sr.clone(), dim=(-1, -2, -3), keepdim=True
    ) ** (1 / 5)
    pixel_level_weight = get_local_weights(residual_sr.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_sr < residual_ema] = 0

    return overall_weight
