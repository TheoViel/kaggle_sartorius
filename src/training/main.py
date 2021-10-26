import gc
import torch
from sklearn.model_selection import StratifiedKFold


from model_zoo.models import define_model
from training.train import fit

from data.preparation import prepare_data
from data.transforms import define_pipelines
from data.dataset import SartoriusDataset

from inference.predict import predict
from inference.post_process import preds_to_instance, remove_padding

from utils.torch import seed_everything, count_parameters, save_model_weights
from utils.metrics import iou_map

from params import CELL_TYPES, ORIG_SIZE


def train(config, df_train, df_val, pipelines, fold, log_folder=None):
    """
    Trains a model.
    TODO

    Args:
        config (Config): Parameters.
        dataset (torch Dataset): whole dataset InMemory
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        SegmentationMeter: Meter.
        pandas dataframe: Training history.
        torch model: Trained segmentation model.
    """
    seed_everything(config.seed)

    model = define_model(config.model_config).to(config.device)
    model.zero_grad()

    n_parameters = count_parameters(model)

    train_dataset = SartoriusDataset(
        df_train,
        transforms=pipelines['train'],
    )
    val_dataset = SartoriusDataset(
        df_val,
        transforms=pipelines['val'],
    )

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    fit(
        model,
        train_dataset,
        val_dataset,
        optimizer_name=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        use_fp16=config.use_fp16,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        name = f"{config.decoder}_{config.encoder}_{fold}.pt"
        save_model_weights(
            model,
            name,
            cp_folder=log_folder,
        )

    return model


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
        activations=config.activations,
        batch_size=config.val_bs,
        use_tta=config.use_tta,
        device=config.device
    )

    return results

    # preds = remove_padding(preds, ORIG_SIZE)

    # pred_cell_types = [CELL_TYPES[i] for i in preds_cls.argmax(-1)]

    # preds_instance = preds_to_instance(preds, pred_cell_types)

    # truths = [m[..., 0] for m in dataset.masks]
    # score = iou_map(truths, preds_instance)

    # print(f' -> Validation IoU mAP = {score:.3f}')

    # return preds, preds_instance, truths


def k_fold(config, log_folder=None):
    """
    Performs a  k-fold cross validation.
    TODO

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """

    df = prepare_data()

    skf = StratifiedKFold(n_splits=config.k, shuffle=True, random_state=config.random_state)
    splits = list(skf.split(X=df, y=df["cell_type"]))

    all_results = []

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            pipelines = define_pipelines(config.data_config)

            model = train(config, df_train, df_val, pipelines, i, log_folder=log_folder)

            results = validate(df_val, model, config, pipelines)

            return results
            all_results += results

            if log_folder is None or len(config.selected_folds) == 1:
                break

            del (model, results)
            torch.cuda.empty_cache()
            gc.collect()

    # cv_score = iou_map(all_truths, all_preds_instance, verbose=0)
    # print(f'\n -> CV IoU mAP : {cv_score:.3f}')

    return all_results
