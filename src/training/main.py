import gc
import glob
import torch
import numpy as np
from sklearn.model_selection import GroupKFold

from training.train import fit
from model_zoo.models import define_model

from data.preparation import get_plate_wells
from data.transforms import get_transfos, get_transfos_inference
from data.dataset import SartoriusDataset
from inference.predict import predict
from utils.torch import seed_everything, count_parameters, save_model_weights
from utils.metrics import dice_score


def train(config, paths_train, paths_val, fold, log_folder=None):
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
        num_classes_cls=config.num_classes_cls,
        encoder_weights=config.encoder_weights,
        reduce_stride=config.reduce_stride,
    ).to(config.device)
    model.zero_grad()

    n_parameters = count_parameters(model)

    transforms_train = get_transfos(mean=model.mean, std=model.std)
    transforms_val = get_transfos(augment=False, mean=model.mean, std=model.std)

    train_dataset = SartoriusDataset(
        paths_train,
        transforms=transforms_train,
    )
    val_dataset = SartoriusDataset(
        paths_val,
        transforms=transforms_val,
    )

    val_dataset_ = SartoriusDataset(paths_val)
    ref_dice = 0
    for i in range(len(val_dataset)):
        img, mask, y = val_dataset_[i]
        # print(img[1:2].shape, img[1:2].max(), mask.shape)
        ref_dice += dice_score(img[:, :, 1][None] > (0.45 * 255), mask[None])

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters")
    print(f"    -> Reference dice : {ref_dice / len(val_dataset) :.4f}\n")

    preds_mask, preds_cls = fit(
        model,
        train_dataset,
        val_dataset,
        loss_config=config.loss_config,
        activations=config.activations,
        optimizer_name=config.optimizer,
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

    return preds_mask, preds_cls


def validate(paths, model, config):
    """
    Validation on full images.
    TODO

    Args:
        model (torch model): Trained model.
        config (Config): Model config.
    """
    dataset = SartoriusDataset(
        paths,
        transforms=get_transfos_inference(mean=model.mean, std=model.std),
    )

    preds_mask, preds_cls = predict(
        dataset,
        model,
        activations=config.activations,
        batch_size=config.val_bs,
        use_tta=config.use_tta,
        device=config.device
    )

    return preds_mask, preds_cls


def k_fold(config, seg_fold=0, log_folder=None):
    """
    Performs a  k-fold cross validation.
    TODO

    Args:
        config (Config): Parameters.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """

    paths = np.array(glob.glob(config.root + f"{seg_fold}_*"))
    plate_wells = get_plate_wells(paths)
    n_splits = len(set(plate_wells))

    skf = GroupKFold(n_splits=n_splits)
    splits = list(skf.split(X=paths, groups=plate_wells))

    all_preds_mask, all_preds_cls = [], []

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {n_splits}  -------------\n")

            paths_train = paths[train_idx]
            paths_val = paths[val_idx]

            assert len(
                set(get_plate_wells(paths_val)).intersection(set(get_plate_wells(paths_train)))
            ) == 0, "Leak"

            preds_mask, preds_cls = train(config, paths_train, paths_val, i, log_folder=log_folder)

            all_preds_mask.append(preds_mask)
            all_preds_cls.append(preds_cls)

            torch.cuda.empty_cache()
            gc.collect()

    return all_preds_mask, all_preds_cls
