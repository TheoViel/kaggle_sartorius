import numpy as np
import pandas as pd

from training.train import fit
from model_zoo.models import define_model
from data.dataset import SartoriusDataset
from data.transforms import get_transfos
from data.preparation import prepare_data, get_splits, prepare_extra_data
from utils.torch import seed_everything, count_parameters, save_model_weights, freeze_batchnorm


def train(
    config, df_train, df_val, fold, log_folder=None
):
    """
    Trains a model

    Args:
        config (Config): Config.
        df_train (pandas DataFrame): Training metadata.
        df_val (pandas DataFrame): Validation metadata.
        pipelines (dict): Augmentation pipelines.
        fold (int): Fold number.
        log_folder (str, optional): Folder to log results to. Defaults to None.
        precompute_masks (bool, optional): Whether to precompute masks. Defaults to True.
        df_extra (pandas DataFrame, optional): Extra metadata. Defaults to None.

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
    """
    seed_everything(config.seed)

    model = define_model(
        config.encoder,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
    ).to(config.device)
    model.zero_grad()

    if config.freeze_bn:
        freeze_batchnorm(model)
    n_parameters = count_parameters(model)

    train_dataset = SartoriusDataset(
        df_train,
        transforms=get_transfos(size=config.size, augment=True, mean=model.mean, std=model.std),
    )

    val_dataset = SartoriusDataset(
        df_val,
        transforms=get_transfos(size=config.size, augment=False, mean=model.mean, std=model.std),
    )

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    preds_cell, preds_plate = fit(
        model,
        train_dataset,
        val_dataset,
        optimizer_name=config.optimizer,
        scheduler_name=config.scheduler,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        verbose_eval=config.verbose_eval,
        first_epoch_eval=config.first_epoch_eval,
        compute_val_loss=config.compute_val_loss,
        num_classes=config.num_classes,
        use_extra_samples=config.use_pl,
        freeze_bn=False,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        name = f"{config.encoder}_{fold}.pt"
        save_model_weights(model, name, cp_folder=log_folder)

    return preds_cell, preds_plate


def k_fold(config, log_folder=None):
    """
    Performs a  k-fold cross validation.

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
    """
    df = prepare_data(fix=False, remove_anomalies=config.remove_anomalies)
    df_extra = prepare_extra_data()
    df = pd.concat([df, df_extra]).reset_index(drop=True)

    splits = get_splits(df, config)

    preds_cell_oof = np.zeros((len(df), config.num_classes))
    preds_plate_oof = np.zeros((len(df), config.num_classes_aux))

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            preds_cell, preds_plate = train(
                config, df_train, df_val, i, log_folder=log_folder
            )

            preds_cell_oof[val_idx] = preds_cell
            preds_plate_oof[val_idx] = preds_plate

            if len(config.selected_folds) == 1:
                return preds_cell, preds_plate

    if log_folder is not None:
        np.save(log_folder + "preds_cell_oof.npy", preds_cell_oof)
        np.save(log_folder + "preds_plate_oof.npy", preds_plate_oof)

    return preds_cell_oof, preds_plate_oof
