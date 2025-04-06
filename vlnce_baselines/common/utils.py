from typing import Any, Dict, List
import torch
import torch.distributed as dist
import numpy as np
import copy

def extract_instruction_tokens(
    observations: List[Dict],
    instruction_sensor_uuid: str,
    tokens_uuid: str = "tokens",
) -> Dict[str, Any]:
    r"""Extracts instruction tokens from an instruction sensor if the tokens
    exist and are in a dict structure.
    """
    for i in range(len(observations)):
        if (
            isinstance(observations[i][instruction_sensor_uuid], dict)
            and tokens_uuid in observations[i][instruction_sensor_uuid]
        ):
            observations[i][instruction_sensor_uuid] = observations[i][
                instruction_sensor_uuid
            ]["tokens"]
        else:
            break
    return observations

def gather_list_and_concat(list_of_nums,world_size):
    if not torch.is_tensor(list_of_nums):
        tensor = torch.Tensor(list_of_nums).cuda()
    else:
        if list_of_nums.is_cuda == False:
            tensor = list_of_nums.cuda()
        else:
            tensor = list_of_nums
    gather_t = [torch.ones_like(tensor) for _ in
                range(world_size)]
    dist.all_gather(gather_t, tensor)
    return gather_t

def dis_to_con(path, amount=0.25):
    starts = path[:-1]
    ends = path[1:]
    new_path = [path[0]]
    for s, e in zip(starts,ends):
        vec = np.array(e) - np.array(s)
        ratio = amount/np.linalg.norm(vec[[0,2]])
        unit = vec*ratio
        times = int(1/ratio)
        for i in range(times):
            if i != times - 1:
                location = np.array(new_path[-1])+unit
                new_path.append(location.tolist())
        new_path.append(e)
    
    return new_path