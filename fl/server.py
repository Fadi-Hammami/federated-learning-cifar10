"""
Server-side pipeline (FedLab "Handler").

FedAvgServerHandler extends FedLab's SyncServerHandler:
  * downlink_package : what the server broadcasts each round  -> [global_parameters]
  * sample_clients   : random subset of clients for the round (sample_ratio)
  * load(payload)    : buffers each client's uplink package; when all expected
                       packages have arrived it calls global_update()
  * global_update    : FedAvg aggregation, weighted by each client's number of
                       training samples  (w_k = n_k / sum n_k)
  * evaluate         : accuracy / loss of the global model on the CIFAR-10 test set,
                       stored in self.history for plotting
"""
import time
import torch
import torch.nn.functional as F

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool


class FedAvgServerHandler(SyncServerHandler):
    def __init__(self, model, global_round, sample_ratio, test_loader=None,
                 cuda=False, device=None, logger=None):
        super().__init__(model, global_round, sample_ratio, cuda, device, logger)
        self.test_loader = test_loader
        self.history = {"round": [], "test_acc": [], "test_loss": [],
                        "update_norm_std": [], "time": []}
        self._t0 = time.time()

    # ---- aggregation --------------------------------------------------------------
    def global_update(self, buffer):
        """buffer = list of client packages, each = [serialized_params, n_samples]."""
        params_list = [pack[0] for pack in buffer]
        weights = torch.tensor([float(pack[1]) for pack in buffer])

        # client drift indicator: how far each local model moved from the global one
        global_params = self.model_parameters
        norms = torch.stack([(p - global_params).norm() for p in params_list])
        self.history["update_norm_std"].append(norms.std().item() if len(norms) > 1 else 0.0)

        new_params = Aggregators.fedavg_aggregate(params_list, weights)
        SerializationTool.deserialize_model(self._model, new_params)

    def load(self, payload):
        finished = super().load(payload)   # returns True when the round is complete
        if finished:
            self.evaluate()
        return finished

    # ---- evaluation ---------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self):
        if self.test_loader is None:
            return None
        acc, loss = evaluate_model(self._model, self.test_loader, self.device if self.cuda else "cpu")
        self.history["round"].append(self.round)
        self.history["test_acc"].append(acc)
        self.history["test_loss"].append(loss)
        self.history["time"].append(time.time() - self._t0)
        self._LOGGER.info(f"Round {self.round:3d} | test loss {loss:.4f} | test acc {acc*100:.2f}%")
        return acc, loss


@torch.no_grad()
def evaluate_model(model, loader, device="cpu"):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total, loss_sum / total
