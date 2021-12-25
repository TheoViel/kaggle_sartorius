from utils.metrics import dice_score_tensor


class SegmentationMeter:
    """
    Meter to handle predictions & metrics.
    """
    def __init__(self, threshold=0.5):
        """
        Constructor

        Args:
            threshold (float, optional): Threshold for predictions. Defaults to 0.5.
        """
        self.threshold = threshold
        self.reset()

    def update(self, pred_mask, pred_cls, y_mask, y_cls):
        """
        Updates the metric.

        Args:
            y_batch (tensor): Truths.
            preds (tensor): Predictions.

        Raises:
            NotImplementedError: Mode not implemented.
        """
        self.dice += dice_score_tensor(
            pred_mask, y_mask, threshold=self.threshold
        ) * pred_mask.size(0)

        self.tps += ((pred_cls.view(y_cls.size()) > 0.5) == y_cls).sum().item()
        self.count += pred_mask.size(0)

    def compute(self):
        """
        Computes the metrics.

        Returns:
            dict: Metrics dictionary.
        """
        self.metrics["dice"] = self.dice / self.count
        self.metrics["acc"] = self.tps / self.count
        return self.metrics

    def reset(self):
        """
        Resets everything.
        """
        self.dice = 0
        self.count = 0
        self.tps = 0
        self.metrics = {
            "dice": 0,
            "acc": 0,
        }
        return self.metrics
