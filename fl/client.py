"""
Client-side pipeline (FedLab "Trainer").

FedAvgSerialTrainer extends FedLab's SGDSerialClientTrainer, which simulates
several clients one after the other inside a single process:

  * local_process(payload, id_list) : receives the global parameters and the list
                                      of client ids to simulate; trains each one
  * train(params, loader)           : local SGD for `epochs` epochs on the client data
  * uplink_package                  : list of packages, one per trained client:
                                      [local_parameters, n_samples]

The same trainer is reused in cross-process mode (each process owns a group
of client ids).
"""
import torch

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer


class FedAvgSerialTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None):
        super().__init__(model, num_clients, cuda, device, logger)
        self.local_losses = {}     # cid -> last local training loss (for the variance study)

    def setup_optim(self, epochs, batch_size, lr, momentum=0.0):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr, momentum=momentum)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id_list):
        global_params = payload[0]
        for cid in id_list:
            loader = self.dataset.get_dataloader(cid, self.batch_size)
            pack = self.train(global_params, loader, cid)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader, cid=None):
        self.set_model(model_parameters)          # start from the global model
        self._model.train()
        n_samples, loss_sum, n_batches = 0, 0.0, 0
        for _ in range(self.epochs):
            for x, y in train_loader:
                if self.cuda:
                    x, y = x.cuda(self.device), y.cuda(self.device)
                out = self._model(x)
                loss = self.criterion(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item(); n_batches += 1
        n_samples = len(train_loader.dataset)
        if cid is not None:
            self.local_losses[cid] = loss_sum / max(n_batches, 1)
        return [self.model_parameters, torch.tensor(float(n_samples))]
