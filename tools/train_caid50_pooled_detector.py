from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.CAIDBenchmarkProtocol import _load_protocol, _protocol_tasks, _task_raw_task_ids
from datasets.safe_sample import make_bad_sample, safe_collate_drop_bad
from utils.train_utils import select_model


CAIDBENCH_MEAN = (0.485, 0.456, 0.406)
CAIDBENCH_STD = (0.229, 0.224, 0.225)


class ArrowIndexDataset(Dataset):
    def __init__(self, root, frame, transform=None, image_column="image"):
        self.root = Path(root).expanduser()
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_column = image_column
        self._readers = {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        try:
            row = self.frame.iloc[int(index)]
            image = self._read_image(row)
            if self.transform is not None:
                image = self.transform(image)
            label = int(row["label"])
            task_id = int(row["task_id"])
            generator_name = str(row["generator_name"])
            return image, label, task_id, generator_name
        except Exception as exc:
            return make_bad_sample(index, exc)

    def _read_image(self, row):
        import pyarrow as pa
        import pyarrow.ipc as ipc

        full_path = Path(str(row["arrow_path"])).expanduser()
        if not full_path.is_absolute():
            full_path = self.root / full_path
        key = str(full_path)
        if key not in self._readers:
            source = pa.memory_map(key, "r")
            self._readers[key] = (source, ipc.open_file(source))
        reader = self._readers[key][1]
        batch = reader.get_batch(int(row["batch_id"]))
        payload = batch.column(self.image_column)[int(row["row_in_batch"])].as_py()
        return Image.open(BytesIO(payload)).convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one pooled CAID detector without protocol stage 0 and report per-generator train/test gap."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--protocol", default="protocol_presets/caidbench/model_appearance_order_protocol_50.yaml")
    parser.add_argument("--index_path", default=None)
    parser.add_argument("--output_dir", default="outputs/caid50_pooled_detector")
    parser.add_argument("--backbone", default="vit_base_patch16_224")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--train_eval_max_per_generator", type=int, default=2000)
    parser.add_argument("--test_eval_max_per_generator", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def setup_logging(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
    )


def resolve_index(protocol_path, index_path):
    if index_path:
        return Path(index_path).expanduser()
    protocol = _load_protocol(protocol_path)
    raw_path = protocol.get("index_path")
    if raw_path is None:
        raise ValueError(f"Protocol has no index_path: {protocol_path}")
    raw_path = Path(raw_path).expanduser()
    return raw_path if raw_path.is_absolute() else Path(protocol_path).resolve().parent / raw_path


def protocol_task_ids(protocol_path):
    protocol = _load_protocol(protocol_path)
    tasks = _protocol_tasks(protocol)
    entries = []
    for stage_id, task in enumerate(tasks):
        name = str(task.get("name", task.get("id", f"task{stage_id}")))
        for task_id in _task_raw_task_ids(task):
            entries.append((stage_id, name, int(task_id)))
    return entries


def build_frames(args):
    protocol_path = Path(args.protocol).expanduser()
    if not protocol_path.is_absolute():
        protocol_path = ROOT / protocol_path
    index_path = resolve_index(protocol_path, args.index_path)
    index = pd.read_parquet(index_path)
    task_entries = protocol_task_ids(protocol_path)
    selected_task_ids = [task_id for stage_id, _, task_id in task_entries if stage_id != 0]
    name_by_task_id = {task_id: name for stage_id, name, task_id in task_entries if stage_id != 0}

    frame = index[index["task_id"].isin(selected_task_ids)].copy()
    frame["generator_name"] = frame["task_id"].map(name_by_task_id).fillna(frame["generator_name"])

    train = frame[frame["split"] == "train"].copy()
    test = frame[frame["split"] == "test"].copy()
    if args.max_train_samples and args.max_train_samples > 0 and len(train) > args.max_train_samples:
        train = train.sample(n=args.max_train_samples, random_state=args.seed).reset_index(drop=True)
    return train.reset_index(drop=True), test.reset_index(drop=True), task_entries


def transforms_for(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(CAIDBENCH_MEAN, CAIDBENCH_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(CAIDBENCH_MEAN, CAIDBENCH_STD),
        ]
    )
    return train_transform, eval_transform


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metric_dict(targets, scores, preds):
    result = {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
    }
    if len(set(targets)) >= 2:
        result["ap"] = float(average_precision_score(targets, scores))
        result["auc"] = float(roc_auc_score(targets, scores))
    else:
        result["ap"] = None
        result["auc"] = None
    return result


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, num_workers, max_per_generator=0, amp=False):
    if max_per_generator and max_per_generator > 0:
        grouped = defaultdict(list)
        for idx, task_id in enumerate(dataset.frame["task_id"].astype("int64").tolist()):
            grouped[int(task_id)].append(idx)
        selected = []
        rng = random.Random(20260625)
        for indices in grouped.values():
            if len(indices) > max_per_generator:
                indices = rng.sample(indices, max_per_generator)
            selected.extend(indices)
        selected.sort()
        eval_dataset = Subset(dataset, selected)
    else:
        eval_dataset = dataset

    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=safe_collate_drop_bad,
    )
    model.eval()
    all_targets = []
    all_scores = []
    all_preds = []
    per_generator = defaultdict(lambda: {"targets": [], "scores": [], "preds": []})

    for batch in loader:
        if batch is None:
            continue
        images, labels, task_ids, generator_names = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        targets = labels.detach().cpu().numpy().astype(int).tolist()
        scores = probs.detach().cpu().numpy().astype(float).tolist()
        pred_values = preds.detach().cpu().numpy().astype(int).tolist()
        all_targets.extend(targets)
        all_scores.extend(scores)
        all_preds.extend(pred_values)

        for name, target, score, pred in zip(generator_names, targets, scores, pred_values):
            bucket = per_generator[str(name)]
            bucket["targets"].append(target)
            bucket["scores"].append(score)
            bucket["preds"].append(pred)

    overall = metric_dict(all_targets, all_scores, all_preds)
    per_generator_metrics = {
        name: {
            **metric_dict(values["targets"], values["scores"], values["preds"]),
            "n": len(values["targets"]),
        }
        for name, values in sorted(per_generator.items())
    }
    return {"overall": {**overall, "n": len(all_targets)}, "per_generator": per_generator_metrics}


def add_gap(train_metrics, test_metrics):
    rows = {}
    names = sorted(set(train_metrics["per_generator"]) | set(test_metrics["per_generator"]))
    for name in names:
        train_row = train_metrics["per_generator"].get(name, {})
        test_row = test_metrics["per_generator"].get(name, {})
        row = {
            "train": train_row,
            "test": test_row,
            "gap": {},
        }
        for key in ["accuracy", "f1", "ap", "auc"]:
            train_value = train_row.get(key)
            test_value = test_row.get(key)
            row["gap"][key] = (
                float(train_value - test_value)
                if train_value is not None and test_value is not None
                else None
            )
        rows[name] = row
    return rows


def save_table(rows, path):
    flat_rows = []
    for name, payload in rows.items():
        row = {"generator": name}
        for split in ["train", "test"]:
            for key, value in payload[split].items():
                row[f"{split}_{key}"] = value
        for key, value in payload["gap"].items():
            row[f"gap_{key}"] = value
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(path, index=False)


def log_gap_table(epoch, rows):
    logging.info(
        "per-generator gap epoch=%s | generator | train_acc | test_acc | gap_acc | train_ap | test_ap | gap_ap | train_auc | test_auc | gap_auc | train_n | test_n",
        epoch,
    )
    for name, payload in rows.items():
        train_row = payload["train"]
        test_row = payload["test"]
        gap = payload["gap"]
        logging.info(
            "per-generator gap epoch=%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s",
            epoch,
            name,
            _fmt_optional(train_row.get("accuracy")),
            _fmt_optional(test_row.get("accuracy")),
            _fmt_optional(gap.get("accuracy")),
            _fmt_optional(train_row.get("ap")),
            _fmt_optional(test_row.get("ap")),
            _fmt_optional(gap.get("ap")),
            _fmt_optional(train_row.get("auc")),
            _fmt_optional(test_row.get("auc")),
            _fmt_optional(gap.get("auc")),
            train_row.get("n", 0),
            test_row.get("n", 0),
        )


def _fmt_optional(value):
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    setup_logging(output_dir)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("device=%s", device)
    train_frame, test_frame, task_entries = build_frames(args)
    logging.info(
        "loaded train=%s test=%s protocol_tasks=%s excluding protocol stage 0",
        len(train_frame),
        len(test_frame),
        len(task_entries),
    )
    logging.info("train generators=%s", sorted(train_frame["generator_name"].unique().tolist()))

    train_transform, eval_transform = transforms_for(args.image_size)
    train_dataset = ArrowIndexDataset(args.data_root, train_frame, transform=train_transform)
    train_eval_dataset = ArrowIndexDataset(args.data_root, train_frame, transform=eval_transform)
    test_dataset = ArrowIndexDataset(args.data_root, test_frame, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=safe_collate_drop_bad,
    )

    model = select_model(
        "slca",
        args.backbone,
        num_classes=2,
        n_tasks=1,
        kwargs={"pretrained": not args.no_pretrained},
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = torch.nn.CrossEntropyLoss()

    best_test_auc = -math.inf
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch_idx, batch in enumerate(train_loader, start=1):
            if batch is None:
                continue
            images, labels, _, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += int(labels.numel())
            loss_sum += float(loss.item()) * int(labels.numel())
            pred = torch.argmax(logits.detach(), dim=-1)
            correct += int((pred == labels).sum().item())
            seen += int(labels.numel())
            if batch_idx % 200 == 0:
                logging.info(
                    "epoch=%s batch=%s/%s seen=%s loss=%.4f train_batch_acc=%.4f lr=%.6g",
                    epoch,
                    batch_idx,
                    len(train_loader),
                    global_step,
                    loss_sum / max(1, seen),
                    correct / max(1, seen),
                    optimizer.param_groups[0]["lr"],
                )

        logging.info(
            "epoch=%s done train_loss=%.4f train_acc_running=%.4f",
            epoch,
            loss_sum / max(1, seen),
            correct / max(1, seen),
        )

        if args.eval_every_epoch > 0 and epoch % args.eval_every_epoch == 0:
            train_metrics = evaluate(
                model,
                train_eval_dataset,
                device,
                args.batch_size,
                args.num_workers,
                max_per_generator=args.train_eval_max_per_generator,
                amp=args.amp,
            )
            test_metrics = evaluate(
                model,
                test_dataset,
                device,
                args.batch_size,
                args.num_workers,
                max_per_generator=args.test_eval_max_per_generator,
                amp=args.amp,
            )
            rows = add_gap(train_metrics, test_metrics)
            payload = {
                "epoch": epoch,
                "global_step": global_step,
                "train_eval": train_metrics,
                "test_eval": test_metrics,
                "per_generator_gap": rows,
                "args": vars(args),
            }
            history.append(payload)
            (output_dir / "latest_metrics.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            save_table(rows, output_dir / "latest_per_generator_gap.csv")
            test_auc = test_metrics["overall"].get("auc")
            train_acc = train_metrics["overall"].get("accuracy")
            test_acc = test_metrics["overall"].get("accuracy")
            logging.info(
                "eval epoch=%s train_acc=%.4f train_auc=%s test_acc=%.4f test_auc=%s",
                epoch,
                train_acc,
                train_metrics["overall"].get("auc"),
                test_acc,
                test_auc,
            )
            log_gap_table(epoch, rows)
            if test_auc is not None and test_auc > best_test_auc:
                best_test_auc = test_auc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                        "test_auc": test_auc,
                    },
                    output_dir / "best.pt",
                )
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                output_dir / "latest.pt",
            )

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logging.info("done output_dir=%s", output_dir)


if __name__ == "__main__":
    main()
