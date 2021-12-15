from collections import OrderedDict
import torch
import torch.distributed as dist


def custom_parse_losses(losses, current_epoch, max_epoch):
    """
    https://github.com/open-mmlab/mmdetection/blob/f3817df55709e4b13098ebaa9cf72a77b68ce994/mmdet/models/detectors/base.py#L176
    Parse the raw outputs (losses) of the network.
    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.
        current_epoch (int) : current training epoch
        max_epoch (int) : maximum number of epochs
    Returns:
        tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
            which may be a weighted sum of all losses, log_vars contains \
            all the variables to be sent to the logger.
    """


    weights = {"loss_mask" : 1.0,
               "loss_cls" : (max_epoch + 1 - current_epoch) / (max_epoch + 1),
               "loss_bbox" : (max_epoch + 1 - current_epoch //2) / (max_epoch + 1),
               "loss_rpn_cls" :(max_epoch + 1 - current_epoch //2) / (max_epoch + 1),
               "loss_rpn_bbox" :(max_epoch + 1 - current_epoch//2) / (max_epoch + 1),
    }

    possible_keys = [_key for _key, _value in weights.items()
                 if 'loss' in _key]

    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = 0
    sum_weights = 0
    for _key, _value in log_vars.items():
        # pick the corresponding decay key
        corresponding_key = [k for k in possible_keys if k in _key]
        if len(corresponding_key)==1:
            loss += _value * weights[corresponding_key[0]]
            sum_weights += weights[corresponding_key[0]] 
        # if weights.get(_key) is not None:
        #     loss += _value * weights[_key]
            # if _key=="loss_cls":
            #     loss += (max_epoch - current_epoch) * _value / max_epoch
            # else:
            #     loss += _value
    # Normalize the loss to avoid extra learning rate decay
    loss /= sum_weights
    # loss = sum(_value for _key, _value in log_vars.items()
    #             if 'loss' in _key)
    # If the loss_vars has different length, GPUs will wait infinitely
    if dist.is_available() and dist.is_initialized():
        log_var_length = torch.tensor(len(log_vars), device=loss.device)
        dist.all_reduce(log_var_length)
        message = (f'rank {dist.get_rank()}' +
                    f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                    ','.join(log_vars.keys()))
        assert log_var_length == len(log_vars) * dist.get_world_size(), \
            'loss log variables are different across GPUs!\n' + message

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars