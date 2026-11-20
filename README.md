# Mini-projet — Apprentissage fédéré avec FedLab (CIFAR-10, FedAvg, IID vs non-IID)

```
fl_cifar/
├── fl/
│   ├── dataset.py      # FedCIFAR10 : partition IID / non-IID (FedLab CIFAR10Partitioner), get_dataloader(cid)
│   ├── model.py        # CNN_CIFAR10 (modèle global, 620 362 paramètres)
│   ├── server.py       # FedAvgServerHandler : sample, broadcast, FedAvg pondéré, évaluation
│   └── client.py       # FedAvgSerialTrainer : SGD local, upload [params, n_samples]
├── standalone.py       # simulation standalone (serveur + clients dans 1 processus)
├── cross_process/
│   ├── server.py       # processus serveur (rank 0)  — torch.distributed / gloo
│   ├── client.py       # processus client  (rank ≥ 1) — chaque processus simule un groupe de clients
│   ├── run_local.sh    # lance 1 serveur + 2 clients (Linux / WSL / macOS)
│   └── run_local.bat   # idem pour Windows (3 fenêtres) — voir le guide : sous Windows préférer WSL
├── download_data.py    # télécharge CIFAR-10 (170 Mo) dans data/cifar10
├── plot_results.py     # figures accuracy/loss, distribution, variance, drift + tableau récap
├── results/            # *.json + logs
└── figures/            # *.png générés
```

## Installation (Windows, PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install fedlab==1.3.0 numpy matplotlib
python download_data.py
```

## Expériences standalone (une commande par ligne, sans retour à la ligne)
```powershell
$env:PYTHONUTF8=1
python standalone.py --partition iid --rounds 40 --tag A_iid
python standalone.py --partition noniid-shards --rounds 40 --tag B_noniid_shards
python standalone.py --partition noniid-dir --dir_alpha 0.3 --rounds 40 --tag B_noniid_dir
python plot_results.py
```
Défauts : 20 clients, 10 clients/round, 2 epochs locales, batch 64, lr 0,05, seed 0.

## Cross-process (1 serveur + 2 processus clients)
Linux / WSL : `bash cross_process/run_local.sh iid 5 CP_iid`
À la main, 3 terminaux, serveur d'abord :
```
python cross_process/server.py --world_size 3 --rounds 5 --partition iid --tag CP_iid
python cross_process/client.py --world_size 3 --rank 1 --partition iid
python cross_process/client.py --world_size 3 --rank 2 --partition iid
```
