import torch

from mmcv.parallel import MMDataParallel

from model_zoo.models import define_model
from model_zoo.ensemble import EnsembleModel

from data.preparation import get_splits
from data.loader import define_loaders
from data.dataset import SartoriusDataset
from data.transforms import define_pipelines

from utils.torch import load_model_weights
from inference.predict import predict


def inference_val(df, configs, weights, ens_config, verbose=1, cell_type=None):
    """
    Inference on the validation data.

    Args:
        df (pandas DataFrame): Metadata.
        configs (list of Config): Model configs.
        weights (list of list of strings): Model weights.
        ens_config (dict): Parameters for the ensemble model.
        verbose (int, optional): Verbosity. Defaults to 1.
        cell_type (str, optional): Cells to consider. Defaults to None.

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
        list of pandas DataFrame: Validation DataFrames.
    """
    pipelines = define_pipelines(configs[0].data_config)

    models = []
    for config in configs:
        model = define_model(config.model_config, encoder=config.encoder, verbose=0)
        models.append(model)

    splits = get_splits(df, configs[0])

    all_results, dfs = [], []
    for i, (train_idx, val_idx) in enumerate(splits):
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)

        if cell_type is not None:
            df_val = df_val[df_val['cell_type'] == cell_type].reset_index(drop=True)

        dfs.append(df_val)

        dataset = SartoriusDataset(
            df_val, transforms=pipelines['test_tta'] if ens_config['use_tta'] else pipelines['test']
        )

        models_trained, names = [], []
        usage = {"cort": [], "astro": [], "shsy5y": [], "cls": []}

        for model_idx, model in enumerate(models):
            weight = weights[model_idx][i]
            assert weight.endswith(f'_{i}.pt'), "Wrong model weights"
            models_trained.append(load_model_weights(model, weight, verbose=verbose))

            names.append(weight.split('/')[-1])

            if ens_config["use_for_cort"][model_idx]:
                usage["cort"].append(model_idx)
            if ens_config["use_for_astro"][model_idx]:
                usage["astro"].append(model_idx)
            if ens_config["use_for_shsy5y"][model_idx]:
                usage["shsy5y"].append(model_idx)
            if ens_config["use_for_cls"][model_idx]:
                usage["cls"].append(model_idx)

        if verbose:
            for c in usage.keys():
                print(f'Models used for {c} :\n', [names[idx] for idx in usage[c]])
            print()

        model = MMDataParallel(
            EnsembleModel(
                models_trained,
                ens_config,
                names=names,
                usage=usage,
            )
        )

        results = predict(dataset, model, batch_size=1, device=config.device)
        all_results.append(results)

        if not all([len(w) > i + 1 for w in weights]):  # Missing weights
            break

    return all_results, dfs


def inference_single(df, configs, weights, ens_config, cell_type=None, idx=0):
    """
    Inference on a single image from the first fold.
    TODO

    Args:
        df (pandas DataFrame): Metadata.
        configs (list of Config): Model configs.
        weights (list of list of strings): Model weights.
        ens_config (dict): Parameters for the ensemble model.
        idx (int, optional): Image index.

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
        tuple: Intermediate outputs from the model.
        list of pandas DataFrame: Validation DataFrames.
    """
    pipelines = define_pipelines(configs[0].data_config)

    models = []
    for config in configs:
        model = define_model(config.model_config, encoder=config.encoder, verbose=0)
        models.append(model)

    splits = get_splits(df, configs[0])

    for i, (train_idx, val_idx) in enumerate(splits):
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)

        if cell_type is not None:
            df_val = df_val[df_val['cell_type'] == "cort"]
        df = df_val.head(idx + 1).tail(1).reset_index(drop=True)

        models_trained, names = [], []
        usage = {"cort": [], "astro": [], "shsy5y": [], "cls": []}

        for model_idx, model in enumerate(models):
            weight = weights[model_idx][i]
            assert weight.endswith(f'_{i}.pt'), "Wrong model weights"
            models_trained.append(load_model_weights(model, weight, verbose=0))

            names.append(weight.split('/')[-1])

            if ens_config["use_for_cort"][model_idx]:
                usage["cort"].append(model_idx)
            if ens_config["use_for_astro"][model_idx]:
                usage["astro"].append(model_idx)
            if ens_config["use_for_shsy5y"][model_idx]:
                usage["shsy5y"].append(model_idx)
            if ens_config["use_for_cls"][model_idx]:
                usage["cls"].append(model_idx)

        model = MMDataParallel(
            EnsembleModel(
                models_trained,
                ens_config,
                names=names,
                usage=usage,
            )
        )

        dataset = SartoriusDataset(
            df,
            transforms=pipelines['test_tta'] if ens_config['use_tta'] else pipelines['test'],
        )

        # predict
        loader = define_loaders(None, dataset, val_bs=1, num_workers=0)[1]
        model.eval()
        with torch.no_grad():
            for batch in loader:
                (boxes, masks), all_stuff = model(
                    **batch, return_loss=False, rescale=True, return_everything=True
                )
                results = [(boxes.cpu().numpy(), masks.cpu().numpy())]

        return results, all_stuff, df
