import gc
import time
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score

from data.loader import define_loaders
from training.optim import define_optimizer, define_scheduler
from utils.torch import freeze_batchnorm


def fit(
    model,
    train_dataset,
    val_dataset,
    optimizer_name="Adam",
    scheduler_name="linear",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    weight_decay=0,
    verbose=1,
    verbose_eval=5,
    first_epoch_eval=0,
    compute_val_loss=True,
    use_fp16=False,
    num_classes=3,
    use_extra_samples=False,
    freeze_bn=False,
    device="cuda",
):
    """
    Training function.

    Args:
        model (torch Model): Model to train.
        train_dataset (SartoriusDataset): Training dataset.
        val_dataset (SartoriusDataset): Validation dataset.
        predict_dataset (SartoriusDataset): Validation dataset for prediction.
        optimizer_name (str, optional): Optimizer name. Defaults to "Adam".
        scheduler_name (str, optional): Scheduler name. Defaults to "linear".
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        weight_decay (int, optional): Weight decay. Defaults to 0.
        verbose (int, optional): Epoch proportion to display logs at. Defaults to 1.
        verbose_eval (int, optional): Epoch proportion to validate at. Defaults to 5.
        first_epoch_eval (int, optional): Epoch to start validating at. Defaults to 0.
        compute_val_loss (bool, optional): Whether to compute the validation loss. Defaults to True.
        num_classes (int, optional): Number of classes. Defaults to 3.
        use_extra_samples (bool, optional): Whether to use extra samples. Defaults to False.
        freeze_bn (bool, optional): Whether to freeze batchnorm layers. Defaults to False.
        device (str, optional): Training device. Defaults to "cuda".

    Returns:
        list of tuples: Results in the MMDet format [(boxes, masks), ...].
    """
    dt = 0.

    optimizer = define_optimizer(
        optimizer_name, model, lr=lr, weight_decay=weight_decay
    )

    train_loader, val_loader = define_loaders(
        train_dataset, val_dataset, batch_size=batch_size, val_bs=val_bs
    )

    num_training_steps = (
        len(train_dataset.img_paths) * epochs
    ) // batch_size
    num_warmup_steps = int(warmup_prop * num_training_steps)
    scheduler = define_scheduler(scheduler_name, optimizer, num_warmup_steps, num_training_steps)

    ce_loss = nn.CrossEntropyLoss(reduction="none")

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_bn:
            freeze_batchnorm(model)
        start_time = time.time()
        optimizer.zero_grad()
        avg_loss = 0

        for img, y_cell, y_plate in train_loader:
            pred_cell, pred_plate = model(img.cuda())

            loss = 0.5 * (
                ce_loss(pred_cell, y_cell.cuda()) +
                ce_loss(pred_plate, y_plate.cuda())
            ).mean()

            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        model.eval()
        avg_val_loss, cell_acc, plate_acc = 0, 0, 0

        do_eval = (epoch >= first_epoch_eval and not epoch % verbose_eval) or (epoch == epochs)
        if do_eval:
            preds_cell, preds_plate = [], []
            with torch.no_grad():
                for img, y_cell, y_plate in val_loader:
                    pred_cell, pred_plate = model(img.cuda())
                    loss = 0.5 * (
                        ce_loss(pred_cell.detach(), y_cell.cuda()) +
                        ce_loss(pred_plate.detach(), y_plate.cuda())
                    ).mean()
                    avg_val_loss += loss.item() / len(val_loader)

                    pred_cell = pred_cell.softmax(-1).detach().cpu().numpy()
                    pred_plate = pred_plate.softmax(-1).detach().cpu().numpy()
                    preds_cell.append(pred_cell)
                    preds_plate.append(pred_plate)

            preds_cell = np.concatenate(preds_cell, 0)
            preds_plate = np.concatenate(preds_plate, 0)
            cell_acc = accuracy_score(val_dataset.y_cell, preds_cell.argmax(-1))
            plate_acc = accuracy_score(val_dataset.y_plate, preds_plate.argmax(-1))

        # Print infos
        dt += time.time() - start_time
        lr = scheduler.get_last_lr()[0]

        string = f"Epoch {epoch:02d}/{epochs:02d} \t lr={lr:.1e}\t t={dt:.0f}s\tloss={avg_loss:.3f}"
        string = string + f"\t avg_val_loss={avg_val_loss:.3f}" if avg_val_loss else string
        string = string + f"\t cell_acc={cell_acc:.3f}" if cell_acc else string
        string = string + f"\t plate_acc={plate_acc:.3f}" if plate_acc else string

        if verbose:
            print(string)
            dt = 0

        del (loss, img, y_cell, y_plate)
        gc.collect()
        torch.cuda.empty_cache()

    return preds_cell, preds_plate
