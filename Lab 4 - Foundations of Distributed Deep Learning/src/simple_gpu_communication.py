import os
import torch
from torch import distributed as dist

# 1
rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"I am rank {rank} with a world size of {world_size} and a GPU of {torch.cuda.get_device_name(0)}")

# 2
dist.init_process_group("nccl")

# 3
torch.cuda.set_device(local_rank)
message = [None]
if rank == 0:
    message = [torch.Tensor([1,2,3,4]).cuda()]

dist.broadcast_object_list(message, src=0)
print(f"I am rank {rank} and the tensor I have is --> {message[0]}")


dist.destroy_process_group()
