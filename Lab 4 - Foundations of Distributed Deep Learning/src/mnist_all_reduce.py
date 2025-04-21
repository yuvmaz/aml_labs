import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from torchvision.datasets import MNIST
from torchvision import transforms

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
print(f"I am rank {rank}")

dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

# Get data

# 1
if local_rank == 0:
    print(f"Downloading MNIST on rank {rank}")
    MNIST(root=".", train=True, download=True)
dist.barrier(device_ids=[local_rank])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
    ])

# 2
dataset = MNIST(root='.', train=True, download=False, transform=transform)
sampler = torch.utils.data.DistributedSampler(dataset)
loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1, 
                                     shuffle=False)

loader_iter = iter(loader)
images, labels = loader_iter.__next__()

images = images.cuda()
labels = labels.cuda()

# 3
print(f"Rank {rank}, label value: {labels}")

# Run neural network

model = nn.Sequential(
    nn.Linear(28*28, 300),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Softmax(dim=-1),
)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad()
predictions = model(images.view(1,28*28))
print(predictions)
loss = F.nll_loss(predictions, labels)
loss.backward()
optimizer.step()

# 4
print("Rank {}, Values before all reduce:{}".format(rank, model[0].weight.data[0,:10]))

# 5
for parameter in model.parameters():
    dist.all_reduce(parameter, op=dist.ReduceOp.AVG)

# 6
print("\n-----\nRank {}, Values after all reduce:{}".format(rank, model[0].weight.data[0,:10]))

dist.destroy_process_group()
