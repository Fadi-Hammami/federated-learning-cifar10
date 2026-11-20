"""
Cross-process FL — CLIENT process (rank 1 .. world_size-1).

Each client process simulates a *group* of clients (a FedLab SerialClientTrainer)
and waits passively for the server (PassiveClientManager):
    receive [client ids, global params] -> local training -> upload packages.

Run one process per rank, e.g. with world_size=3 (2 client processes):
  python cross_process/client.py --world_size 3 --rank 1 --partition iid
  python cross_process/client.py --world_size 3 --rank 2 --partition iid
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fedlab.core.network import DistNetwork
from fedlab.core.client.manager import PassiveClientManager
from fedlab.utils.logger import Logger

from fl import FedCIFAR10, CNN_CIFAR10, FedAvgSerialTrainer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--port", default="3002")
    p.add_argument("--world_size", type=int, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--partition", default="iid", choices=["iid", "noniid-shards", "noniid-dir"])
    p.add_argument("--num_clients", type=int, default=20, help="total simulated clients (same on every process)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ethernet", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    torch.manual_seed(args.seed + args.rank)

    # Every process rebuilds the SAME partition (same seed) and owns a contiguous
    # block of client ids: n_proc = world_size-1 processes, each ~num_clients/n_proc clients.
    dataset = FedCIFAR10(args.data_root, args.num_clients, partition=args.partition, seed=args.seed)
    n_proc = args.world_size - 1
    per_proc = args.num_clients // n_proc
    extra = args.num_clients - per_proc * n_proc
    my_clients = per_proc + (1 if args.rank <= extra else 0)   # reported to the server's Coordinator

    model = CNN_CIFAR10()
    trainer = FedAvgSerialTrainer(model, num_clients=my_clients, logger=Logger(log_name=f"client-{args.rank}"))
    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    network = DistNetwork(address=(args.ip, args.port), world_size=args.world_size, rank=args.rank,
                          ethernet=args.ethernet)
    manager = PassiveClientManager(network=network, trainer=trainer, logger=Logger(log_name=f"manager-{args.rank}"))
    print(f"[client rank {args.rank}] simulating {my_clients} clients, waiting for server at {args.ip}:{args.port}")
    manager.run()
    print(f"[client rank {args.rank}] finished.")
