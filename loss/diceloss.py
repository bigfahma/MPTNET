import torch
def DiceScore(
    y_pred: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Computes Dice score metric from full size Tensor and collects average.
    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
    Returns:
        Dice scores per batch and per class, (shape [batch_size, num_classes]).
    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    denominator = y_o + y_pred_o

    return torch.where(
        denominator > 0,
        (2.0 * intersection) / denominator,
        torch.tensor(float("1"), device=y_o.device),
    )


