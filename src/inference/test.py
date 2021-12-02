# OUTDATED

from inference.predict import predict
from data.dataset import SartoriusInferenceDataset
from data.transforms import define_pipelines
from model_zoo.ensemble import EnsembleModel
from model_zoo.models import define_model
from utils.torch import load_model_weights
from mmcv.parallel import MMDataParallel


def inference_test(df, configs, weights, use_tta=False):

    pipelines = define_pipelines(configs[0].data_config)

    models = []
    for config, fold_weights in zip(configs, weights):
        for weight in fold_weights:
            model = define_model(
                config.model_config, encoder=config.encoder, verbose=0
            )
            model = load_model_weights(model, weight)
            models.append(model)

    dataset = SartoriusInferenceDataset(
        df, transforms=pipelines['test_tta'] if use_tta else pipelines['test']
    )

    model = MMDataParallel(EnsembleModel(models))

    results = predict(
        dataset,
        model,
        batch_size=1,
        device=config.device
    )

    return results
