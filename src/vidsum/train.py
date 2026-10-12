"""Train one (config, dataset, sigma, split) run with CA-SUM's protocol.

    python -m vidsum.train --config configs/casum_googlenet.yaml --split 0 --reg_factor 0.6

The test set is never used to choose anything here. Test scores are written for every epoch (as CA-SUM
does) and the checkpoint kept is the one `evaluate.py` will select: minimum training loss after
`select.start_epoch`. Writes to experiments/<exp>/<dataset>/reg<sigma>/split<k>/.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data.dataset import build_loaders
from .models.ca_sum import MultimodalCASUM, init_weights
from .objectives.losses import unsupervised_loss
from .paths import EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--reg_factor", type=float, help="Override train.reg_factor (sigma)")
    parser.add_argument("--dataset", help="Override dataset")
    parser.add_argument("--n_epochs", type=int, help="Override train.n_epochs (e.g. for smoke tests)")
    parser.add_argument("--exp", help="Experiment name; defaults to the config file name")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def load_config(args):
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.reg_factor is not None:
        cfg["train"]["reg_factor"] = args.reg_factor
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.n_epochs is not None:
        cfg["train"]["n_epochs"] = args.n_epochs
    cfg["exp"] = args.exp or cfg.get("exp") or Path(args.config).stem
    cfg["split"] = args.split
    return cfg


def run_dir_for(cfg):
    return EXPERIMENTS / cfg["exp"] / cfg["dataset"] / f"reg{cfg['train']['reg_factor']}" / f"split{cfg['split']}"


def select_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(item, device):
    return (
        {m: f.to(device) for m, f in item["features"].items()},
        {m: f.to(device) for m, f in item["masks"].items()},
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores = {}
    for item in loader:
        feats, masks = to_device(item, device)
        out, _, _ = model(feats, masks)
        scores[item["key"]] = out.cpu().numpy().tolist()
    return scores


def main():
    args = parse_args()
    cfg = load_config(args)
    tcfg, mcfg = cfg["train"], cfg["model"]
    device = select_device(args.device)

    run_dir = run_dir_for(cfg)
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    train_loader, test_loader = build_loaders(cfg["dataset"], cfg["split"], cfg["modalities"])
    set_seed(tcfg["seed"])
    model = MultimodalCASUM(
        train_loader.dataset.dims,
        d_model=mcfg.get("d_model"),
        fusion=mcfg.get("fusion", "concat"),
        block_size=mcfg.get("block_size", 60),
        heads=mcfg.get("heads", 8),
    ).to(device)
    if mcfg.get("init_type"):
        init_weights(model, mcfg["init_type"], mcfg.get("init_gain", 1.4142))
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["l2_req"])

    print(f"[{cfg['exp']}] {cfg['dataset']} split {cfg['split']} sigma={tcfg['reg_factor']} "
          f"modalities={cfg['modalities']} params={sum(p.numel() for p in model.parameters()):,} device={device}")

    def save_scores(epoch):
        with open(run_dir / "scores" / f"epoch_{epoch}.json", "w", encoding="utf-8") as handle:
            json.dump(predict(model, test_loader, device), handle)

    save_scores(-1)  # untrained model, needed by the label-free sigma selection

    start_epoch = cfg.get("select", {}).get("start_epoch", 20)
    history, best_loss = [], float("inf")
    batch_size = tcfg["batch_size"]
    t0 = time.time()
    for epoch in range(tcfg["n_epochs"]):
        model.train()
        losses, terms_log = [], []
        # Like CA-SUM: floor(N / batch_size) accumulated batches per epoch; leftover videos are skipped.
        num_batches = max(1, len(train_loader) // batch_size)
        iterator = iter(train_loader)
        for _ in range(num_batches):
            optimizer.zero_grad()
            for _ in range(min(batch_size, len(train_loader))):
                feats, masks = to_device(next(iterator), device)
                scores, _, _ = model(feats, masks)
                loss, terms = unsupervised_loss(scores, tcfg)
                loss.backward()
                losses.append(float(loss.detach()))
                terms_log.append(terms)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip"])
            optimizer.step()

        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": epoch_loss,
                        **{k: float(np.mean([t[k] for t in terms_log])) for k in terms_log[0]}})
        save_scores(epoch)
        if epoch >= start_epoch and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), run_dir / "selected.pt")
        if epoch % 50 == 0 or epoch == tcfg["n_epochs"] - 1:
            print(f"  epoch {epoch:4d}  loss {epoch_loss:.6f}  ({time.time() - t0:.0f}s)")

    torch.save(model.state_dict(), run_dir / "last.pt")
    with open(run_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump({"train_seconds": time.time() - t0, "epochs": history}, handle, indent=2)
    print(f"  done in {time.time() - t0:.0f}s -> {run_dir}")


if __name__ == "__main__":
    main()
