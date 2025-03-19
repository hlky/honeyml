import torch

def get_device_name():
    cleanup = {
        "nvidia ": "",
        "geforce rtx ": "",
        "geforce gtx ": "",
        "geforce gt ": "",
        "geforce ": "",
        "tesla ": "",
        "quadro ": "",
        " ": "_",
    }
    split_by = {
        ",": 0,
        "(": 0,
    }

    device_name = torch.cuda.get_device_name().lower()
    for target, replacement in cleanup.items():
        device_name = device_name.replace(target, replacement).strip()
    
    for target, index in split_by.items():
        device_name = device_name.split(target)[index].strip()
    
    return device_name

def get_sm():
    sm = "".join(str(i) for i in torch.cuda.get_device_capability())
    return sm
