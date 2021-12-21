from sklearn.model_selection import StratifiedKFold

from training.train import fit
from model_zoo.models import define_model
from data.dataset import SartoriusDataset
from data.transforms import define_pipelines
from data.preparation import prepare_data, prepare_extra_data, get_splits, prepare_pl_data
from utils.torch import seed_everything, count_parameters, save_model_weights, freeze_batchnorm


def train(
    config, df_train, df_val, pipelines, fold, log_folder=None, precompute_masks=True, df_extra=None
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
        config.model_config,
        encoder=config.encoder,
        pretrained_livecell=config.pretrained_livecell,
    ).to(config.device)
    model.zero_grad()

    if config.freeze_bn:
        freeze_batchnorm(model)
    n_parameters = count_parameters(model)

    train_dataset = SartoriusDataset(
        df_train,
        transforms=pipelines['train'],
        precompute_masks=precompute_masks,
        df_extra=df_extra,
    )

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
    print(f"    -> {n_parameters} trainable parameters")

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
        use_extra_samples=config.use_extra_samples or config.use_pl,
        freeze_bn=config.freeze_bn,
        device=config.device,
        loss_decay=config.loss_decay,
        config=config,
        log_folder=log_folder,
        fold=fold,
    )

    if config.save_weights and log_folder is not None:
        name = f"{config.name}_{config.encoder}_{fold}.pt"
        save_model_weights(model, name, cp_folder=log_folder)

    return results


def k_fold(config, log_folder=None):
    """
    Performs a  k-fold cross validation.

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
    """
    df_extra = None
    df = prepare_data(fix=False, remove_anomalies=config.remove_anomalies)
    df_fix = prepare_data(fix=True, remove_anomalies=config.remove_anomalies)

    if config.use_extra_samples:
        assert not config.use_pl, "Cannot use PL and extra data"
        df_extra = prepare_extra_data(config.extra_name)

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

            if config.use_pl:
                df_extra = prepare_pl_data(f"pl_ensnew_{i}", fold=i, verbose=1)

                for plate_well in df_extra['plate_well'].unique():
                    assert plate_well not in df_val['plate_well'].values, f"{plate_well} in val."

            pipelines = define_pipelines(config.data_config)

            results = train(
                config, df_train, df_val, pipelines, i, log_folder=log_folder, df_extra=df_extra
            )
            all_results += results

            if log_folder is None or len(config.selected_folds) == 1:
                return results

    return all_results


def pretrain(config, log_folder=None):
    """
    Pretrains a model.

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """
    df = prepare_extra_data(name="livecell")

    skf = StratifiedKFold(n_splits=config.k, shuffle=True, random_state=config.random_state)
    splits = list(skf.split(X=df, y=df["cell_type"]))

    for i, (train_idx, val_idx) in enumerate(splits):
        df_train = df.iloc[train_idx].copy().reset_index(drop=True)
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)

        pipelines = define_pipelines(config.data_config)

        results = train(
            config, df_train, df_val, pipelines, i, log_folder=log_folder, precompute_masks=False
        )

        return results
