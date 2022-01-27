"""
Original Source: 
    https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py
    https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py

Modified according to customized vision_transfomer.py modules
"""
import json
import math


def param_groups_lrd(
    model,
    weight_decay=0.05,
    no_weight_decay_list=[],
    layer_decay=0.75,
    verbose=False,
):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.encoder.layers) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    if verbose:
        print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ["cls_token", "pos_embed"]:
        return 0
    elif name.startswith(
        "patch_embedding"
    ):  # e.g. "patch_embedding.project_patch.weight"
        return 0
    elif name.startswith("encoder") and name.split(".")[2].isdigit():
        return (
            int(name.split(".")[2]) + 1
        )  # e.g. "encoder.layers.1.self_attn.qkv.weights"
    else:
        return num_layers  # e.g. "head.weight"


def adjust_learning_rate(optimizer, epoch, lr_config={}):
    """Decay the learning rate with half-cycle cosine after warmup"""

    # parameters
    base_lr = lr_config.get("lr", 1e-3)
    min_lr = lr_config.get("min_lr", 1e-6)
    warmup_epochs = lr_config.get("warmup_epochs", 5)
    total_epochs = lr_config.get("epochs", 50)

    # LR scheduling
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0
            + math.cos(
                math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )

    # update lr within optimizer's parameter groups
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
