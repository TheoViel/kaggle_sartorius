from inference.predict import predict
from data.dataset import SartoriusDataset, SartoriusInferenceDataset
from data.transforms import define_pipelines
from utils.metrics import evaluate_results
from model_zoo.models import define_model
from utils.torch import load_model_weights


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


def inference_test(df, config, weights):

    pipelines = define_pipelines(config.data_config)

    model = define_model(
        config.model_config, reduce_stride=config.reduce_stride, pretrained_weights=None
    )

    dataset = SartoriusInferenceDataset(df, transforms=pipelines['test'])

    all_results = []
    for i, weight in enumerate(weights):
        model = load_model_weights(model, weight)

        results = predict(
            dataset,
            model,
            batch_size=1,
            use_tta=config.use_tta,
            device=config.device
        )
        all_results.append(results)

    return all_results
