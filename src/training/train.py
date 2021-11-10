import gc
import time
import torch
import traceback
from tqdm.notebook import tqdm  # noqa

from data.loader import define_loaders
from training.optim import define_optimizer, define_scheduler
from inference.predict import predict
from utils.metrics import evaluate_results


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
    device="cuda",
):
    """
    Usual torch fit function.
    TODO

    Args:
        model (torch model): Model to train.
        dataset (InMemoryTrainDataset): Dataset.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        activations (str, optional): Activation functions. Defaults to 'sigmoid'.
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        mix_proba (float, optional): Probability to apply mixup with. Defaults to 0.
        mix_alpha (float, optional): Mixup alpha parameter. Defaults to 0.4.
        verbose (int, optional): Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional): Epoch to start evaluating at. Defaults to 0.
        num_classes (int, optional): Number of classes. Defaults to 1.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(val_dataset) x num_classes]: Last prediction on the validation data.
        pandas dataframe: Training history.
    """
    dt = 0.

    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(
        optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay
    )

    train_loader, val_loader = define_loaders(
        train_dataset, val_dataset, batch_size=batch_size, val_bs=val_bs
    )

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = define_scheduler(scheduler_name, optimizer, num_warmup_steps, num_training_steps)

    for epoch in range(1, epochs+1):
        model.train()
        start_time = time.time()
        optimizer.zero_grad()
        avg_loss = 0

        for batch in train_loader:
            if use_fp16:  # TODO
                with torch.cuda.amp.autocast():
                    losses = model(**batch, return_loss=True)
                    loss, _ = model.module._parse_losses(losses)

                    scaler.scale(loss).backward()
                    avg_loss += loss.item() / len(train_loader)

                    # TODO : grad clip
                    scaler.step(optimizer)

                    scale = scaler.get_scale()
                    scaler.update()

                    if scale == scaler.get_scale():
                        scheduler.step()

            else:
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
                predict_dataset, model, batch_size=1, device=device
            )
            try:
                iou_map, _ = evaluate_results(predict_dataset, results)
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
