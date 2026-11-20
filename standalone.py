"""
Standalone federated simulation on CIFAR-10 (1 server + N clients in one process).

Usage examples:
  python standalone.py --partition iid           --rounds 40 --tag A_iid
  python standalone.py --partition noniid-shards --rounds 40 --tag B_noniid_shards
  python standalone.py --partition noniid-dir    --rounds 40 --tag B_noniid_dir --dir_alpha 0.3

Outputs results/<tag>.json with per-round accuracy/loss and per-client statistics.
"""
import argparse, json, os, time, random
import numpy as np
import torch

from fedlab.core.standalone import StandalonePipeline
from fedlab.utils.logger import Logger

from fl import FedCIFAR10, CNN_CIFAR10, count_params, FedAvgServerHandler, FedAvgSerialTrainer, evaluate_model


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--partition", default="iid", choices=["iid", "noniid-shards", "noniid-dir"])
    p.add_argument("--shards_per_client", type=int, default=2, help="shards per client for noniid-shards (~classes per client)")
    p.add_argument("--dir_alpha", type=float, default=0.3, help="Dirichlet alpha for noniid-dir")
    p.add_argument("--num_clients", type=int, default=20)
    p.add_argument("--sample_ratio", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--epochs", type=int, default=2, help="local epochs")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default=None)
    p.add_argument("--cuda", action="store_true")
    return p.parse_args()


class Pipeline(StandalonePipeline):
    """FedLab's StandalonePipeline already runs: sample -> broadcast -> local train -> load/aggregate.
    We only override evaluate() (the handler evaluates itself inside load())."""
    def evaluate(self):
        pass


def per_client_stats(model, dataset, device):
    """Accuracy of the final global model on each client's own data + on the test
    samples belonging to the client's classes (measures fairness / variance between clients)."""
    from torch.utils.data import DataLoader, TensorDataset
    out = []
    for cid in range(dataset.num_clients):
        acc_local, _ = evaluate_model(model, dataset.get_dataloader(cid, 500), device)
        classes = dataset.client_classes(cid)
        mask = torch.isin(dataset.test_y, torch.tensor(classes))
        loader = DataLoader(TensorDataset(dataset.test_x[mask], dataset.test_y[mask]), batch_size=1000)
        acc_test, _ = evaluate_model(model, loader, device)
        out.append({"cid": cid, "n_samples": len(dataset.client_dict[cid]), "classes": classes,
                    "acc_on_local_data": acc_local, "acc_on_test_of_own_classes": acc_test})
    return out


def main():
    args = parse()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    tag = args.tag or f"{args.partition}"
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs("results", exist_ok=True)

    # 1) data + partition
    dataset = FedCIFAR10(args.data_root, args.num_clients, partition=args.partition,
                         shards_per_client=args.shards_per_client, dir_alpha=args.dir_alpha, seed=args.seed)
    hist = dataset.class_histogram()
    print(f"[{tag}] partition={args.partition}  clients={args.num_clients}  "
          f"sizes min/max={min(dataset.client_sizes())}/{max(dataset.client_sizes())}")
    for cid in range(min(5, args.num_clients)):
        print(f"   client {cid}: n={hist[cid].sum():5d}  classes={dataset.client_classes(cid)}  hist={hist[cid].tolist()}")

    # 2) global model + server handler + client trainer
    model = CNN_CIFAR10()
    print(f"[{tag}] CNN params: {count_params(model):,}")
    handler = FedAvgServerHandler(model, global_round=args.rounds, sample_ratio=args.sample_ratio,
                                  test_loader=dataset.get_test_loader(), cuda=device == "cuda",
                                  logger=Logger(log_name="server"))
    trainer = FedAvgSerialTrainer(model, args.num_clients, cuda=device == "cuda")
    trainer.setup_dataset(dataset)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr)

    # 3) FedAvg rounds
    t0 = time.time()
    Pipeline(handler, trainer).main()
    elapsed = time.time() - t0

    # 4) per-client statistics of the final global model
    stats = per_client_stats(handler.model, dataset, device)
    accs_local = np.array([s["acc_on_local_data"] for s in stats])
    accs_test = np.array([s["acc_on_test_of_own_classes"] for s in stats])
    local_losses = [trainer.local_losses.get(c, None) for c in range(args.num_clients)]
    local_losses_arr = np.array([l for l in local_losses if l is not None])

    results = {
        "tag": tag, "config": vars(args), "elapsed_sec": elapsed,
        "history": handler.history,
        "final_test_acc": handler.history["test_acc"][-1],
        "class_histogram": hist.tolist(),
        "per_client": stats,
        "client_variance": {
            "acc_local_mean": accs_local.mean(), "acc_local_std": accs_local.std(),
            "acc_local_min": accs_local.min(), "acc_local_max": accs_local.max(),
            "acc_testclasses_mean": accs_test.mean(), "acc_testclasses_std": accs_test.std(),
            "local_loss_mean": float(local_losses_arr.mean()), "local_loss_std": float(local_losses_arr.std()),
        },
    }
    with open(f"results/{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=float)

    # rounds needed to reach accuracy thresholds (convergence speed)
    for th in (0.40, 0.50, 0.60, 0.65, 0.70):
        r = next((rr for rr, a in zip(handler.history["round"], handler.history["test_acc"]) if a >= th), None)
        print(f"[{tag}] rounds to reach {th*100:.0f}% : {r}")
    print(f"[{tag}] final test acc = {results['final_test_acc']*100:.2f}%  "
          f"| per-client acc std = {accs_local.std()*100:.2f} pts | time {elapsed:.0f}s")
    print(f"[{tag}] saved results/{tag}.json")


if __name__ == "__main__":
    main()
