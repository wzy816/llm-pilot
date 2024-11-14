import functools
import os
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import enable_wrap, size_based_auto_wrap_policy, wrap
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

import wandb

# https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# https://arxiv.org/pdf/2304.11277


def count_trainable_parameters(model):
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        total += num_params
        if param.requires_grad:
            trainable += num_params
    return trainable, total, f"{trainable/total:.6f}"


@dataclass
class NetConfig:
    batch_size: int = 64
    test_batch_size: int = 1000
    lr: float = 1.0
    gamma: float = 0.7
    epochs: int = 20
    seed: int = 1
    dt: str = ""
    world_size: int = 0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, rank, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="sum")
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        wandb.log({"train_loss": ddp_loss[0] / ddp_loss[1], "epoch": epoch})


def test(model, rank, test_loader, epoch):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        wandb.log(
            {
                "test_loss": ddp_loss[0] / ddp_loss[2],
                "epoch": epoch,
                "accuracy": ddp_loss[1] / ddp_loss[2],
            }
        )


def fsdp_main(rank, config):
    if rank == 0:
        wandb.init(
            project="fsdp_mnist.py",
            name=config.dt,
            config=config,
            mode="online",
        )
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=config.world_size)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((1.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("./data", train=False, transform=transform)

    sampler1 = DistributedSampler(
        dataset1, rank=rank, num_replicas=config.world_size, shuffle=True
    )
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=config.world_size)

    train_kwargs = {"batch_size": config.batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": config.test_batch_size, "sampler": sampler2}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    torch.cuda.set_device(rank)

    # no wrap policy
    # FullyShardedDataParallel(
    #   (_fsdp_wrapped_module): Net(
    #     (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    #     (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    #     (dropout1): Dropout(p=0.25, inplace=False)
    #     (dropout2): Dropout(p=0.5, inplace=False)
    #     (fc1): Linear(in_features=9216, out_features=128, bias=True)
    #     (fc2): Linear(in_features=128, out_features=10, bias=True)
    #   )
    # )
    # model = Net().to(rank)
    # model = FSDP(model)

    # apply wrap policy
    # FullyShardedDataParallel(
    #   (_fsdp_wrapped_module): Net(
    #     (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
    #     (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    #     (dropout1): Dropout(p=0.25, inplace=False)
    #     (dropout2): Dropout(p=0.5, inplace=False)
    #     (fc1): FullyShardedDataParallel(
    #       (_fsdp_wrapped_module): Linear(in_features=9216, out_features=128, bias=True)
    #     )
    #     (fc2): Linear(in_features=128, out_features=10, bias=True)
    #   )
    # )
    model = Net().to(rank)
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            size_based_auto_wrap_policy, min_num_params=2000
        ),
        # cpu_offload=CPUOffload(offload_params=True),
    )

    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)

    if rank == 0:
        print("training start\n")

    for epoch in range(1, config.epochs + 1):
        train(model, rank, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, test_loader, epoch)
        scheduler.step()

    if rank == 0:
        print("training end\n")
        trainable, total, ratio = count_trainable_parameters(model)
        print(f"{model}\ntrainable: {trainable}, total: {total}, ratio: {ratio}")

    dist.barrier()
    states = model.state_dict()

    if rank == 0:
        print("saving model start")
        torch.save(states, f"./data/fsdp_mnist.{config.dt}.pt")
        print("saving model end")

        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":

    config = NetConfig()
    config.world_size = torch.cuda.device_count()
    config.dt = datetime.now().strftime("%Y%m%d-%H%M%S")

    torch.manual_seed(config.seed)

    mp.spawn(fsdp_main, args=(config,), nprocs=config.world_size, join=True)
