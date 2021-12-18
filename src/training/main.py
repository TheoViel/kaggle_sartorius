import numpy as np
# import pandas as pd

from training.train import fit
from model_zoo.models import define_model
from data.dataset import SartoriusDataset
from data.transforms import get_transfos
from data.preparation import prepare_data, prepare_target, prepare_extra_data
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

    preds = fit(
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

    return preds


def k_fold(config, log_folder=None):
    """
    Performs a  k-fold cross validation.

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
    """
    df = prepare_data(fix=False, remove_anomalies=True)
    df_extra = prepare_extra_data()

    df, df_extra = prepare_target(df, df_extra)

    all_preds = []

    for i in range(config.k):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df['target'] = df[f'target_{i}']
            df_extra['target'] = df_extra[f'target_{i}']

            preds = train(
                config, df, df_extra, i, log_folder=log_folder
            )
            all_preds.append(preds)

    if log_folder is not None:
        np.save(log_folder + "preds.npy", np.array(all_preds))

    return all_preds
