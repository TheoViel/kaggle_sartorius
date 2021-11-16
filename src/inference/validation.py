import torch
import pandas as pd

from mmcv.parallel import MMDataParallel
from sklearn.model_selection import StratifiedKFold

from model_zoo.models import define_model
from model_zoo.ensemble import EnsembleModel
from utils.metrics import evaluate_results
from data.loader import define_loaders
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

    model = define_model(config.model_config, encoder=config.encoder, verbose=0)

    skf = StratifiedKFold(
        n_splits=config.k, shuffle=True, random_state=config.random_state
    )
    splits = list(skf.split(X=df, y=df["cell_type"]))

    all_results, dfs = [], []
    for i, (train_idx, val_idx) in enumerate(splits):
        if len(weights) > i:
            assert weights[i].endswith(f'_{i}.pt'), "Wrong model weights"
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
        model = define_model(config.model_config, encoder=config.encoder, verbose=0)
        models.append(model)

    skf = StratifiedKFold(
        n_splits=configs[0].k, shuffle=True, random_state=configs[0].random_state
    )
    splits = list(skf.split(X=df, y=df["cell_type"]))

    all_results, dfs = [], []
    for i, (train_idx, val_idx) in enumerate(splits):
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)
        dfs.append(df_val)

        models_trained = []
        for model_idx, model in enumerate(models):
            assert weights[model_idx][i].endswith(f'_{i}.pt'), "Wrong model weights"
            models_trained.append(load_model_weights(model, weights[model_idx][i]))

        model = MMDataParallel(EnsembleModel(models_trained))

        results = validate(df_val, model, config, pipelines, use_tta=use_tta)
        all_results += results

        if not all([len(w) > i + 1 for w in weights]):  # Missing weights
            break

    return all_results, pd.concat(dfs).reset_index(drop=True)


def inference_single(df, configs, weights, idx=0, use_tta=False):
    pipelines = define_pipelines(configs[0].data_config)

    models = []
    for config in configs:
        model = define_model(config.model_config, encoder=config.encoder, verbose=0)
        models.append(model)

    skf = StratifiedKFold(
        n_splits=configs[0].k, shuffle=True, random_state=configs[0].random_state
    )
    splits = list(skf.split(X=df, y=df["cell_type"]))

    for i, (train_idx, val_idx) in enumerate(splits):
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)
        df = df_val.head(idx + 1).tail(1).reset_index(drop=True)

        models_trained = []
        for model_idx, model in enumerate(models):
            assert weights[model_idx][i].endswith(f'_{i}.pt'), "Wrong model weights"
            models_trained.append(load_model_weights(model, weights[model_idx][i]))

        model = MMDataParallel(EnsembleModel(models_trained))

        dataset = SartoriusDataset(
            df,
            transforms=pipelines['test_tta'] if use_tta else pipelines['test'],
        )

        # predict
        loader = define_loaders(None, dataset, val_bs=1, num_workers=0)[1]
        model.eval()
        with torch.no_grad():
            for batch in loader:
                results, all_stuff = model(
                    **batch, return_loss=False, rescale=True, return_everything=True
                )
                break

        score, _ = evaluate_results(dataset, results)

        print(f' -> IoU mAP = {score:.3f}')

        return results, all_stuff, df
