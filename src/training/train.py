import gc
import time
import torch
import traceback
import numpy as np

from data.loader import define_loaders
from training.optim import define_optimizer, define_scheduler
from inference.predict import predict
from utils.metrics import quick_eval_results
from utils.torch import freeze_batchnorm


def fit(
    model,
    train_dataset,
    val_dataset,
    predict_dataset,
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

    if use_extra_samples:
        # extra_scheduling = np.clip(  # PL -
        #     [50 * (i ** (1 + 10 / epochs) // 5) for i in range(epochs)][::-1], 0, 1000
        # ).astype(int)
        extra_scheduling = np.clip(  # PL
            [100 * (i ** (1 + 10 / epochs) // 5) for i in range(epochs)][::-1], 0, 1000
        ).astype(int)

        print(f"    -> Extra scheduling : {extra_scheduling.tolist()}\n")
    else:
        extra_scheduling = [0] * epochs
        print()

    num_training_steps = (
        len(train_dataset.img_paths) * epochs + np.sum(extra_scheduling)
    ) // batch_size
    num_warmup_steps = int(warmup_prop * num_training_steps)
    scheduler = define_scheduler(scheduler_name, optimizer, num_warmup_steps, num_training_steps)

    for epoch in range(1, epochs + 1):
        train_dataset.sample_extra_data(extra_scheduling[epoch - 1])
        model.train()
        if freeze_bn:
            freeze_batchnorm(model)
        start_time = time.time()
        optimizer.zero_grad()
        avg_loss = 0

        for batch in train_loader:
            losses = model(**batch, return_loss=True)
            loss, _ = model.module._parse_losses(losses)

            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        model.eval()
        avg_val_loss, iou_map = 0, 0

        do_eval = (epoch >= first_epoch_eval and not epoch % verbose_eval) or (epoch == epochs)
        if do_eval:
            if compute_val_loss:
                with torch.no_grad():
                    for batch in val_loader:
                        losses = model(**batch, return_loss=True)
                        loss, _ = model.module._parse_losses(losses)

                        avg_val_loss += loss.item() / len(val_loader)

            results = predict(
                predict_dataset, model, batch_size=1, device=device, mode="val"
            )
            try:
                iou_map, _ = quick_eval_results(predict_dataset, results, num_classes=num_classes)
            except Exception:
                traceback.print_exc()
                print()
                pass

        # Print infos
        dt += time.time() - start_time
        lr = scheduler.get_last_lr()[0]

        string = f"Epoch {epoch:02d}/{epochs:02d} \t lr={lr:.1e}\t t={dt:.0f}s\tloss={avg_loss:.3f}"
        string = string + f"\t avg_val_loss={avg_val_loss:.3f}" if avg_val_loss else string
        string = string + f"\t iou_map={iou_map:.3f}" if iou_map else string

        if verbose:
            print(string)
            dt = 0

        del (loss, losses, batch)
        gc.collect()
        torch.cuda.empty_cache()

    return results
