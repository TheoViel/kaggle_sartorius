import gc
import torch
from sklearn.model_selection import StratifiedKFold

from training.train import fit
from model_zoo.models import define_model

from data.preparation import prepare_data
from data.transforms import get_transfos, get_transfos_inference
from data.dataset import SartoriusDataset

from inference.predict import predict, get_flip_tta
from inference.post_process import preds_to_instance, remove_padding

from utils.torch import seed_everything, count_parameters, save_model_weights
from utils.metrics import iou_map


def train(config, df_train, df_val, fold, log_folder=None):
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
        config.decoder,
        config.encoder,
        num_classes=config.num_classes,
        encoder_weights=config.encoder_weights,
    ).to(config.device)
    model.zero_grad()

    n_parameters = count_parameters(model)

    transforms_train = get_transfos(
        size=config.size, mean=model.mean, std=model.std
    )
    transforms_val = get_transfos(
        size=config.size_val, augment=False, mean=model.mean, std=model.std
    )

    train_dataset = SartoriusDataset(
        df_train,
        transforms=transforms_train,
    )
    val_dataset = SartoriusDataset(
        df_val,
        transforms=transforms_val,
    )

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    meter, history = fit(
        model,
        train_dataset,
        val_dataset,
        optimizer_name=config.optimizer,
        loss_name=config.loss,
        activation=config.activation,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
        num_classes=config.num_classes,
    )

    if config.save_weights and log_folder is not None:
        name = f"{config.decoder}_{config.encoder}_{fold}.pt"
        save_model_weights(
            model,
            name,
            cp_folder=log_folder,
        )

    return model


def validate(df, model, config):
    """
    Validation on full images.
    TODO

    Args:
        model (torch model): Trained model.
        config (Config): Model config.
    """
    dataset = SartoriusDataset(
        df,
        transforms=get_transfos_inference(mean=model.mean, std=model.std),
    )

    tta_fct = get_flip_tta if config.use_tta else None
    preds = predict(
        dataset, model, tta_fct=tta_fct, activation=config.activation, device=config.device
    )

    truths = [m[..., 0] for m in dataset.masks]

    del dataset
    gc.collect()

    preds = [remove_padding(p, t) for p, t in zip(preds, truths)]

    preds_instance = preds_to_instance(preds)

    score = iou_map(truths, preds_instance)

    print(f'\n -> Validation IoU mAP = {score:.3f}')

    return preds, preds_instance, truths


def k_fold(config, log_folder=None):
    """
    Performs a  k-fold cross validation.
    TODO

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """

    df = prepare_data(width=config.width)

    skf = StratifiedKFold(n_splits=config.k, shuffle=True, random_state=config.random_state)
    splits = list(skf.split(X=df, y=df["cell_type"]))

    all_preds, all_preds_instance, all_truths = [], [], []

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            model = train(config, df_train, df_val, i, log_folder=log_folder)
            preds, preds_instance, truths = validate(df_val, model, config)

            all_preds += preds
            all_preds_instance += preds_instance
            all_truths += truths

            if log_folder is None or len(config.selected_folds) == 1:
                break

            del (model, preds_instance, truths)
            torch.cuda.empty_cache()
            gc.collect()

    return all_preds, all_preds_instance, all_truths