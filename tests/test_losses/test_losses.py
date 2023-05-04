import pytest
import oneflow as flow

from onesr.losses.basic_loss import CharbonnierLoss, L1Loss, MSELoss, WeightedTVLoss


@pytest.mark.parametrize("loss_class", [L1Loss, MSELoss, CharbonnierLoss])
def test_pixellosses(loss_class):
    """Test loss: pixel losses"""

    pred = flow.rand((1, 3, 4, 4), dtype=flow.float32)
    target = flow.rand((1, 3, 4, 4), dtype=flow.float32)
    loss = loss_class(loss_weight=1.0, reduction="mean")
    out = loss(pred, target, weight=None)
    assert isinstance(out, flow.Tensor)
    assert out.shape == flow.Size([])

    # -------------------- test with other reduction -------------------- #
    # reduction = none
    loss = loss_class(loss_weight=1.0, reduction="none")
    out = loss(pred, target, weight=None)
    assert isinstance(out, flow.Tensor)
    assert out.shape == (1, 3, 4, 4)
    # test with spatial weights
    weight = flow.rand((1, 3, 4, 4), dtype=flow.float32)
    out = loss(pred, target, weight=weight)
    assert isinstance(out, flow.Tensor)
    assert out.shape == (1, 3, 4, 4)

    # reduction = sum
    loss = loss_class(loss_weight=1.0, reduction="sum")
    out = loss(pred, target, weight=None)
    assert isinstance(out, flow.Tensor)
    assert out.shape == flow.Size([])

    # -------------------- test unsupported loss reduction -------------------- #
    with pytest.raises(ValueError):
        loss_class(loss_weight=1.0, reduction="unknown")


def test_weightedtvloss():
    """Test loss: WeightedTVLoss"""

    pred = flow.rand((1, 3, 4, 4), dtype=flow.float32)
    loss = WeightedTVLoss(loss_weight=1.0, reduction="mean")
    out = loss(pred, weight=None)
    assert isinstance(out, flow.Tensor)
    assert out.shape == flow.Size([])

    # test with spatial weights
    weight = flow.rand((1, 3, 4, 4), dtype=flow.float32)
    out = loss(pred, weight=weight)
    assert isinstance(out, flow.Tensor)
    assert out.shape == flow.Size([])

    # -------------------- test reduction = sum-------------------- #
    loss = WeightedTVLoss(loss_weight=1.0, reduction="sum")
    out = loss(pred, weight=None)
    assert isinstance(out, flow.Tensor)
    assert out.shape == flow.Size([])

    # test with spatial weights
    weight = flow.rand((1, 3, 4, 4), dtype=flow.float32)
    out = loss(pred, weight=weight)
    assert isinstance(out, flow.Tensor)
    assert out.shape == flow.Size([])

    # -------------------- test unsupported loss reduction -------------------- #
    with pytest.raises(ValueError):
        WeightedTVLoss(loss_weight=1.0, reduction="unknown")
    with pytest.raises(ValueError):
        WeightedTVLoss(loss_weight=1.0, reduction="none")
