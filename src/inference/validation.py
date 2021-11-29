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


def inference_val(df, configs, weights, use_tta=False):

    pipelines = define_pipelines(configs[0].data_config)

    models = []
    for config in configs:
        model = define_model(config.model_config, encoder=config.encoder, verbose=0)
        models.append(model)

    splits = get_splits(df, config)

    all_results, dfs = [], []
    for i, (train_idx, val_idx) in enumerate(splits):
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)
        dfs.append(df_val)

        dataset = SartoriusDataset(
            df_val, transforms=pipelines['test_tta'] if use_tta else pipelines['test']
        )

        models_trained, names = [], []
        for model_idx, model in enumerate(models):
            weight = weights[model_idx][i]
            assert weight.endswith(f'_{i}.pt'), "Wrong model weights"
            models_trained.append(load_model_weights(model, weight))
            names.append(weight.split('/')[-1])

        model = MMDataParallel(
            EnsembleModel(
                models_trained,
                use_tta=use_tta,
                names=names,
                use_tta_proposals=False,
                single_fold_proposals=False
            )
        )

        results = predict(dataset, model, batch_size=1, device=config.device)
        all_results.append(results)

        if not all([len(w) > i + 1 for w in weights]):  # Missing weights
            break

    return all_results, dfs


def inference_single(df, configs, weights, idx=0, use_tta=False):
    pipelines = define_pipelines(configs[0].data_config)

    models = []
    for config in configs:
        model = define_model(config.model_config, encoder=config.encoder, verbose=0)
        models.append(model)

    splits = get_splits(df, config)

    for i, (train_idx, val_idx) in enumerate(splits):
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)
        df = df_val.head(idx + 1).tail(1).reset_index(drop=True)

        models_trained, names = [], []
        for model_idx, model in enumerate(models):
            weight = weights[model_idx][i]
            assert weight.endswith(f'_{i}.pt'), "Wrong model weights"
            models_trained.append(load_model_weights(model, weight))
            names.append(weight.split('/')[-1])

        model = MMDataParallel(
            EnsembleModel(
                models_trained,
                use_tta=use_tta,
                names=names,
                use_tta_proposals=False,
                single_fold_proposals=False
            )
        )

        dataset = SartoriusDataset(
            df,
            transforms=pipelines['test_tta'] if use_tta else pipelines['test'],
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
