import torch.nn as nn
from typing import List
from mmengine.logging import MMLogger

first_set_requires_grad = True
first_set_train = True


def set_requires_grad(model: nn.Module, keywords: List[str]):
    """
    notice:key in name!
    """
    requires_grad_names = []
    num_params = 0
    num_trainable = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        if any(key in name for key in keywords):
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()
        else:
            param.requires_grad = False
    global first_set_requires_grad
    if first_set_requires_grad:
        logger = MMLogger.get_current_instance()
        for name in requires_grad_names:
            logger.info(f"set_requires_grad----{name}")
        logger.info(
            f"Total trainable params--{num_trainable}, All params--{num_params}, Ratio--{num_trainable*100/num_params:.1f}%"
        )
        first_set_requires_grad = False


def _set_train(model: nn.Module, keywords: List[str], prefix: str = ""):
    train_names = []
    for name, child in model.named_children():
        fullname = ".".join([prefix, name])
        if any(name.startswith(key) for key in keywords):
            train_names.append(fullname)
            child.train()
        else:
            train_names += _set_train(child, keywords, prefix=fullname)
    return train_names


def set_train(model: nn.Module, keywords: List[str]):
    """
    notice:sub name startwith key!
    """
    model.train(False)
    train_names = _set_train(model, keywords)
    global first_set_train
    if first_set_train:
        logger = MMLogger.get_current_instance()
        for train_name in train_names:
            logger.info(f"set_train----{train_name}")
        first_set_train = False


def set_requires_grad2(model: nn.Module, keywords: List[str]):
    """
    notice:key in name!
    """
    requires_grad_names = []
    num_params = 0
    num_trainable = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
        if any(key in name for key in keywords):
            param.requires_grad = False
        else:
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()

    global first_set_requires_grad
    if first_set_requires_grad:
        logger = MMLogger.get_current_instance()
        for name in requires_grad_names:
            logger.info(f"set_requires_grad----{name}")
        logger.info(
            f"Total trainable params--{num_trainable}, All params--{num_params}, Ratio--{num_trainable*100/num_params:.1f}%"
        )
        first_set_requires_grad = False

# for lora!!!!
def set_requires_grad3(model: nn.Module, keywords: List[str]):
    """
    notice:key in name!
    """
    requires_grad_names = []
    num_params = 0
    num_trainable = 0

    for name, param in model.named_parameters():
        num_params += param.numel()

        # 如果 name 同时包含 "blocks" 和 "lora"，设置 requires_grad=True
        if "lora" in name:
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()
        # 否则按照 keywords 的逻辑处理
        elif any(key in name for key in keywords):
            param.requires_grad = False
        else:
            param.requires_grad = True
            requires_grad_names.append(name)
            num_trainable += param.numel()

    global first_set_requires_grad
    if first_set_requires_grad:
        logger = MMLogger.get_current_instance()
        for name in requires_grad_names:
            logger.info(f"set_requires_grad----{name}")
        logger.info(
            f"Total trainable params--{num_trainable}, All params--{num_params}, Ratio--{num_trainable*100/num_params:.1f}%"
        )
        first_set_requires_grad = False

def _set_train2(model: nn.Module, keywords: List[str], prefix: str = ""):
    train_names = []
    for name, child in model.named_children():
        fullname = ".".join([prefix, name])
        if not any(name.startswith(key) for key in keywords):
            train_names.append(fullname)
            child.train()
        else:
            train_names += _set_train2(child, keywords, prefix=fullname)
    return train_names

def set_train2(model: nn.Module, keywords: List[str]):
    """
    notice:sub name startwith key!
    """
    model.train(False)
    train_names = _set_train2(model, keywords)
    global first_set_train
    if first_set_train:
        logger = MMLogger.get_current_instance()
        for train_name in train_names:
            logger.info(f"set_train----{train_name}")
        first_set_train = False


def _set_train3(model,keywords):
    # 定义递归函数来遍历子模块
    train_names = []

    def recurse_modules(module, keywords: List[str], parent_name=""):
        for name, submodule in module.named_children():
            # 构建完整的模块名称
            full_name = parent_name + "." + name if parent_name else name

            # 如果子模块名称中包含"lora"，则设置为trainable
            if "lora" in full_name:
                submodule.train()
                train_names.append(full_name)

            # 如果子模块名称中不包含以下关键字，则设置为trainable
            elif not any(x in full_name for x in keywords):
                submodule.train()
                train_names.append(full_name)

            else:
                submodule.eval()  # 其他子模块设置为不参与训练

            # 递归调用以处理子模块的子模块
            recurse_modules(submodule, keywords, full_name)

    # 开始递归遍历模型
    recurse_modules(model,keywords)
    return train_names


def set_train3(model: nn.Module, keywords: List[str]):
    """
    notice:sub name startwith key!
    """
    model.train(False)
    # train_names = _set_train3(model, keywords_yes,keywords_no)
    train_names = _set_train3(model, keywords)
    global first_set_train
    if first_set_train:
        logger = MMLogger.get_current_instance()
        for train_name in train_names:
            logger.info(f"set_train----{train_name}")
        first_set_train = False



