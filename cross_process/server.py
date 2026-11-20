"""
Cross-process FL — SERVER process (rank 0).

The server owns the global CNN and the FedAvg handler. Communication is done
by FedLab's SynchronousServerManager over torch.distributed (gloo backend),
so every client is a separate OS process (or a separate machine).

Run (from the project root):
  python cross_process/server.py --world_size 3 --rounds 20 --partition iid --tag CP_iid

world_size = 1 server + number of client processes.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fedlab.core.network import DistNetwork
from fedlab.core.server.manager import SynchronousServerManager
from fedlab.utils.logger import Logger

from fl import FedCIFAR10, CNN_CIFAR10, FedAvgServerHandler


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--port", default="3002")
    p.add_argument("--world_size", type=int, required=True, help="1 server + N client processes")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--sample_ratio", type=float, default=0.5)
    p.add_argument("--data_root", default="data")
    p.add_argument("--partition", default="iid", choices=["iid", "noniid-shards", "noniid-dir"])
    p.add_argument("--num_clients", type=int, default=20, help="total simulated clients (same on every process)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="cross_process")
    p.add_argument("--ethernet", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse()
    torch.manual_seed(args.seed)

    # the server only needs the test set (for evaluation); partition seed must match the clients
    dataset = FedCIFAR10(args.data_root, args.num_clients, partition=args.partition, seed=args.seed)

    model = CNN_CIFAR10()
    handler = FedAvgServerHandler(model, global_round=args.rounds, sample_ratio=args.sample_ratio,
                                  test_loader=dataset.get_test_loader(), logger=Logger(log_name="server"))

    network = DistNetwork(address=(args.ip, args.port), world_size=args.world_size, rank=0,
                          ethernet=args.ethernet)
    # mode="GLOBAL": client processes receive *global* client ids, which index the shared partition
    manager = SynchronousServerManager(network=network, handler=handler, mode="GLOBAL",
                                       logger=Logger(log_name="server-manager"))
    manager.run()   # setup -> main_loop (rounds) -> shutdown

    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "config": vars(args), "history": handler.history,
                   "final_test_acc": handler.history["test_acc"][-1]}, f, indent=1)
    print(f"[server] done. final test acc = {handler.history['test_acc'][-1]*100:.2f}%  -> results/{args.tag}.json")
