import time
import torch
from transformers import get_linear_schedule_with_warmup

from utils.logger import update_history
from training.meter import SegmentationMeter
from training.loader import define_loaders
from training.optim import define_loss, define_optimizer, prepare_for_loss


def fit(
    model,
    train_dataset,
    val_dataset,
    optimizer_name="Adam",
    loss_name="BCEWithLogitsLoss",
    activation="sigmoid",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    verbose=1,
    first_epoch_eval=0,
    num_classes=1,
    device="cuda",
):
    """
    Usual torch fit function.
    TODO

    Args:
        model (torch model): Model to train.
        dataset (InMemoryTrainDataset): Dataset.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        loss_name (str, optional): Loss name. Defaults to 'BCEWithLogitsLoss'.
        activation (str, optional): Activation function. Defaults to 'sigmoid'.
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
    avg_val_loss = 0.0
    history = None

    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)

    loss_fct = define_loss(loss_name, device=device)

    train_loader, val_loader = define_loaders(train_dataset, val_dataset)

    meter = SegmentationMeter()

    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0

        for batch in train_loader:
            x = batch[0].to(device).float()
            y_batch = batch[1].float()
            y_batch[:, :2] = (y_batch[:, :2] > 0).float()  # ignore instance id

            with torch.cuda.amp.autocast():
                y_pred = model(x)

                y_pred, y_batch = prepare_for_loss(y_pred, y_batch, loss_name, device=device)

                loss = loss_fct(y_pred, y_batch).mean()

                scaler.scale(loss).backward()

                avg_loss += loss.item() / len(train_loader)

                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            for param in model.parameters():
                param.grad = None

        model.eval()
        avg_val_loss = 0.
        metrics = meter.reset()

        if epoch + 1 >= first_epoch_eval:
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device).float()
                    y_batch = batch[1].float()
                    y_batch[:, :2] = (y_batch[:, :2] > 0).float()  # ignore instance id

                    y_pred = model(x)

                    y_pred, y_batch = prepare_for_loss(
                        y_pred,
                        y_batch,
                        loss_name,
                        device=device,
                        train=False
                    )

                    loss = loss_fct(y_pred, y_batch).mean()

                    avg_val_loss += loss / len(val_loader)

                    if activation == "sigmoid":
                        y_pred = torch.sigmoid(y_pred)
                    elif activation == "softmax":
                        y_pred = torch.softmax(y_pred, 2)

                    meter.update(y_batch[:, 0], y_pred[:, 0])

            metrics = meter.compute()

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s\t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )
            if epoch + 1 >= first_epoch_eval:
                print(f"val_loss={avg_val_loss:.3f} \t dice={metrics['dice'][0]:.4f}")
            else:
                print("")
            history = update_history(
                history, metrics, epoch + 1, avg_loss, avg_val_loss, elapsed_time
            )

    del (train_loader, val_loader, y_pred, loss, x, y_batch)
    torch.cuda.empty_cache()

    return meter, history
