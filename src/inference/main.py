from inference.predict import predict
from data.dataset import SartoriusDataset
from utils.metrics import evaluate_results


def validate(df, model, config, pipelines):
    """
    Validation on full images.
    TODO

    Args:
        model (torch model): Trained model.
        config (Config): Model config.
    """
    dataset = SartoriusDataset(
        df,
        transforms=pipelines['test'],
    )

    results = predict(
        dataset,
        model,
        batch_size=1,
        use_tta=config.use_tta,
        device=config.device
    )

    score = evaluate_results(dataset, results)
    print(f' -> Validation IoU mAP = {score:.3f}')

    return results
