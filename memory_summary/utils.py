from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import get_full_repo_name
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model 
from accelerate.utils import get_balanced_memory
import torch
from typing import List

def Qwen_load(model_path, devices: List[int], memory = '35GiB'):
    max_memory = {int(cuda):memory for cuda in devices}
    config = AutoConfig.from_pretrained(model_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype = torch.float16)
    no_split_module_classes = model._no_split_modules
    max_memory = get_balanced_memory(model, max_memory=max_memory)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)

    load_checkpoint_in_model(model, model_path,device_map=device_map) #加载权重
    model = dispatch_model(model, device_map=device_map, skip_keys=model._skip_keys_device_placement) #并分配到具体的设备上
    #skip key必不可少 否则无法正常推理
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained('/nvme/lisongling/models/Qwen1.5-14B-Chat')

    return model, tokenizer