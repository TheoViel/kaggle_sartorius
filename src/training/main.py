from model_zoo.models import define_model
from training.train import fit

from data.preparation import prepare_data, prepare_extra_data, get_splits
from data.transforms import define_pipelines, to_mosaic
from data.dataset import SartoriusDataset

from utils.torch import seed_everything, count_parameters, save_model_weights


def train(
    config, df_train, df_val, pipelines, fold, log_folder=None, precompute_masks=True, df_extra=None
):
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

    model = define_model(
        config.model_config,
        encoder=config.encoder,
        pretrained_livecell=config.pretrained_livecell,
    ).to(config.device)
    model.zero_grad()

    n_parameters = count_parameters(model)

    train_dataset = SartoriusDataset(
        df_train,
        transforms=pipelines['train'],
        precompute_masks=precompute_masks,
        df_extra=df_extra,
    )
    if config.use_mosaic:
        train_dataset = to_mosaic(config, train_dataset)

    val_dataset = SartoriusDataset(
        df_val,
        transforms=pipelines['val'],
        precompute_masks=precompute_masks,
    )
    predict_dataset = SartoriusDataset(
        df_val,
        transforms=pipelines['test'],
        precompute_masks=precompute_masks,
    )

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    results = fit(
        model,
        train_dataset,
        val_dataset,
        predict_dataset,
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
        use_fp16=config.use_fp16,
        use_extra_samples=config.use_extra_samples,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        name = f"{config.name}_{config.encoder}_{fold}.pt"
        save_model_weights(model, name, cp_folder=log_folder)

    return results


def k_fold(config, log_folder=None):
    """
    Performs a  k-fold cross validation.
    TODO

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """

    df = prepare_data(fix=False, remove_anomalies=config.remove_anomalies)
    df_fix = prepare_data(fix=True, remove_anomalies=config.remove_anomalies)

    if config.use_extra_samples > 0:
        df_extra = prepare_extra_data(config.extra_name)
    else:
        df_extra = None

    splits = get_splits(df, config)

    all_results = []

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            if config.fix:
                df_train = df_fix.iloc[train_idx].copy().reset_index(drop=True)
            else:
                df_train = df.iloc[train_idx].copy().reset_index(drop=True)

            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            pipelines = define_pipelines(config.data_config)

            results = train(
                config, df_train, df_val, pipelines, i, log_folder=log_folder, df_extra=df_extra
            )
            all_results += results

            if log_folder is None or len(config.selected_folds) == 1:
                return results

    return all_results
