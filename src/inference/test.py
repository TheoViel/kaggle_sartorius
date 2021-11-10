from inference.predict import predict
from data.dataset import SartoriusInferenceDataset
from data.transforms import define_pipelines
from model_zoo.models import define_model
from utils.torch import load_model_weights


def inference_test(df, config, weights, use_tta=False):

    pipelines = define_pipelines(config.data_config)

    model = define_model(
        config.model_config, encoder=config.encoder
    )

    dataset = SartoriusInferenceDataset(df, transforms=pipelines['test'])

    all_results = []
    for i, weight in enumerate(weights):
        model = load_model_weights(model, weight)

        results = predict(
            dataset,
            model,
            batch_size=1,
            use_tta=use_tta,
            device=config.device
        )
        all_results.append(results)

    return all_results
