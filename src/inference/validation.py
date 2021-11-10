import pandas as pd

from mmcv.parallel import MMDataParallel
from sklearn.model_selection import StratifiedKFold

from model_zoo.models import define_model
from model_zoo.ensemble import EnsembleModel
from utils.metrics import evaluate_results
from data.dataset import SartoriusDataset
from data.transforms import define_pipelines
from utils.torch import load_model_weights
from inference.predict import predict


def validate(df, model, config, pipelines, use_tta=False):
    """
    Validation on full images.
    TODO

    Args:
        model (torch model): Trained model.
        config (Config): Model config.
    """
    dataset = SartoriusDataset(
        df,
        transforms=pipelines['test_tta'] if use_tta else pipelines['test'],
    )

    results = predict(
        dataset,
        model,
        batch_size=1,
        device=config.device
    )

    score, _ = evaluate_results(dataset, results)
    print(f' -> Validation IoU mAP = {score:.3f}')

    return results


def inference_val(df, config, weights):

    pipelines = define_pipelines(config.data_config)

    model = define_model(config.model_config, encoder=config.encoder)

    skf = StratifiedKFold(
        n_splits=config.k, shuffle=True, random_state=config.random_state
    )
    splits = list(skf.split(X=df, y=df["cell_type"]))

    all_results, dfs = [], []
    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)
            dfs.append(df_val)

            model = load_model_weights(model, weights[i])

            results = validate(df_val, model, config, pipelines)
            all_results += results

    return all_results, pd.concat(dfs).reset_index(drop=True)


def inference_val_ens(df, configs, weights, use_tta=False):

    pipelines = define_pipelines(configs[0].data_config)

    models = []
    for config in configs:
        model = define_model(config.model_config, encoder=config.encoder)
        models.append(model)

    skf = StratifiedKFold(
        n_splits=configs[0].k, shuffle=True, random_state=configs[0].random_state
    )
    splits = list(skf.split(X=df, y=df["cell_type"]))

    all_results, dfs = [], []
    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)
            dfs.append(df_val)

            models_trained = []
            for model_idx, model in enumerate(models):
                models_trained.append(load_model_weights(model, weights[model_idx][i]))

            model = MMDataParallel(EnsembleModel(models_trained))

            results = validate(df_val, model, config, pipelines, use_tta=use_tta)
            all_results += results

    return all_results, pd.concat(dfs).reset_index(drop=True)
