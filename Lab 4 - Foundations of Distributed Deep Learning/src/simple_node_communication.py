import os
from torch import distributed as dist

# 1
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"I am rank {rank} with a world size of {world_size}")

# 2
dist.init_process_group("gloo")

#3
message = [None]
if rank == 0:
    message = [f"Hello world from rank {rank}"]

# 4
dist.broadcast_object_list(message, src=0)
print(f"I am rank {rank} and the object I have is --> {message[0]}")

# 5
dist.destroy_process_group()

