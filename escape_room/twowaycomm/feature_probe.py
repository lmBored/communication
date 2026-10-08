"""Frozen-representation ridge probes with pose-disjoint train/val/test sets."""

import copy
import math

import numpy as np
import torch
from tensordict import TensorDict

from escape_room.twowaycomm.scene import (
    ARROW_POSITIONS, ARROW_BASE_YAW_DEG, SENDER_SPAWNS, RECEIVER_SPAWN,
)


def fit_linear_probe(features, labels, pose_ids, seed=20261007):
    """Train-only scaling, validation-only ridge selection, untouched test poses.

    The dual ridge solve avoids a feature-dimensional square matrix for 192px
    flattened CNNs. Confidence intervals resample camera poses, not duplicate
    task conditions. This tests linear accessibility, not absence of all info.
    """
    train, val, test = (features[key].double() for key in ("train", "val", "test"))
    classes = torch.unique(labels["train"], sorted=True)
    centre = train.mean(0)
    scale = train.std(0, unbiased=False)
    keep = scale > 1e-6
    if not keep.any():
        predictions = torch.full_like(labels["test"], classes[0])
        best_alpha = None
        val_accuracy = float((labels["val"] == classes[0]).double().mean())
    else:
        scale = scale[keep].clamp_min(1e-4)
        train = (train[:, keep] - centre[keep]) / scale
        val = (val[:, keep] - centre[keep]) / scale
        test = (test[:, keep] - centre[keep]) / scale
        width = train.shape[-1]
        kernel = train @ train.T / width
        val_kernel, test_kernel = val @ train.T / width, test @ train.T / width
        target = (labels["train"][:, None] == classes[None]).double()
        bias = target.mean(0)
        eye = torch.eye(len(train), device=train.device, dtype=torch.float64)
        best = None
        for alpha in (.0001, .001, .01, .1, 1., 10., 100.):
            weights = torch.linalg.solve(kernel + alpha * eye, target - bias)
            pred = classes[(val_kernel @ weights + bias).argmax(-1)]
            accuracy = float((pred == labels["val"]).double().mean())
            if best is None or accuracy > best[0]:
                best = (accuracy, alpha, weights)
        val_accuracy, best_alpha, weights = best
        predictions = classes[(test_kernel @ weights + bias).argmax(-1)]
    correct = (predictions == labels["test"]).double().cpu().numpy()
    test_poses = np.asarray(pose_ids["test"])
    per_pose = np.array([correct[test_poses == key].mean() for key in np.unique(test_poses)])
    rng = np.random.default_rng(seed)
    boot = rng.choice(per_pose, size=(2000, len(per_pose)), replace=True).mean(-1)
    return {"test_accuracy": float(correct.mean()), "chance": 1 / len(classes),
            "pose_bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
            "validation_accuracy": val_accuracy, "ridge_alpha": best_alpha,
            "input_features": int(features["train"].shape[-1]),
            "varying_features": int(keep.sum()),
            "samples": {key: len(labels[key]) for key in labels},
            "test_pose_count": len(per_pose)}


def random_cnn_control(cnn, seed):
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    result = copy.deepcopy(cnn)
    for module in result.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            module.reset_parameters()
    torch.random.set_rng_state(rng_state)
    return result.eval()


def camera_pose(layout, role, color, regime, rng, broad=False):
    """Physically valid views of the designated sign or doorway, including yaw."""
    if role == "sender":
        if regime == "spawn":
            base = SENDER_SPAWNS[layout]
            yaw = ARROW_BASE_YAW_DEG[layout][color] if layout == "scattered" else 0
        else:
            sign = ARROW_POSITIONS[layout][color]
            yaw = ARROW_BASE_YAW_DEG[layout][color]
            angle = math.radians(yaw)
            base = (sign[0] + 1.7 * math.sin(angle),
                    sign[1] - 1.7 * math.cos(angle), .45)
    else:
        base = RECEIVER_SPAWN if regime == "spawn" else (.5, 2.0, .45)
        yaw = 0
    spread = .20 if broad else .12
    position = torch.tensor([base[0] + rng.uniform(-spread, spread),
                             base[1] + rng.uniform(-spread, spread), .45])
    angle = math.radians(yaw + rng.uniform(-8 if broad else -5, 8 if broad else 5))
    return position, torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])


def collect_probe_dataset(env, actor, conditions, role, color, regime, counts, seed):
    """No actor updates; include raw pixels and untrained CNN controls."""
    term = env.action_manager.get_term("agents")
    term.set_arrow_directions(conditions.directions)
    term.set_door_color(conditions.colors)
    patterns = conditions.condition_id % 8
    if role == "sender":
        ids = torch.stack([(patterns == i).nonzero()[0, 0] for i in range(8)])
        labels = (conditions.directions[ids, color] == 1).long()
        cnn = actor.sender_cnn
    else:
        ids = torch.stack([(conditions.colors == i).nonzero()[0, 0] for i in range(3)])
        labels = conditions.colors[ids]
        cnn = actor.receiver_cnn
    random_cnn = random_cnn_control(cnn, seed + 9001)
    datasets = {name: {} for name in ("trained_cnn", "random_cnn", "raw_pixels")}
    label_sets, pose_sets = {}, {}
    for split_index, (split, count) in enumerate(zip(("train", "val", "test"), counts)):
        rng = np.random.default_rng(seed + 1009 * split_index)
        rows = {name: [] for name in datasets}
        agent_index = 0 if role == "sender" else 1
        for pose_id in range(count):
            position, quat = camera_pose(actor.arrow_layout if hasattr(actor, "arrow_layout") else env.cfg.actions["agents"].arrow_layout,
                                         role, color, regime, rng, broad=split == "test")
            term._set_pose(agent_index, position.to(env.device), None)
            term._sim_data.qpos[:, term._q[agent_index, 3:]] = quat.to(env.device)
            env.sim.forward()
            env.sim.sense()
            image = env.scene[role + "_camera"].data.rgb[ids].permute(0, 3, 1, 2).float() / 255
            rows["trained_cnn"].append(cnn(image).float().cpu())
            rows["random_cnn"].append(random_cnn(image).float().cpu())
            rows["raw_pixels"].append(image.flatten(1).cpu())
        for name in datasets:
            datasets[name][split] = torch.cat(rows[name])
        label_sets[split] = labels.cpu().repeat(count)
        pose_sets[split] = np.repeat(np.arange(count), len(ids))
    del random_cnn
    return datasets, label_sets, pose_sets


def probe_actor(env, actor, conditions, counts=(8, 4, 8), seed=20261007):
    report = {"method": "frozen CNN + train-scaled linear ridge classifier",
              "split": "disjoint train/validation/test camera poses; test pose range is wider",
              "controls": ["untrained CNN with same architecture", "raw RGB pixels"],
              "inference_limit": "low linear-probe accuracy does not prove all information is absent",
              "datasets": {}}
    for regime in ("spawn", "near"):
        for role, color in (("sender", 0), ("sender", 1), ("sender", 2), ("receiver", 0)):
            key = f"{regime}/{role}/" + (f"direction_{color}" if role == "sender" else "door_color")
            datasets, labels, poses = collect_probe_dataset(env, actor, conditions, role, color, regime,
                                                            counts, seed + color * 17)
            report["datasets"][key] = {name: fit_linear_probe(features, labels, poses, seed)
                                        for name, features in datasets.items()}
            print(f"PROBE_PROGRESS {key} trained={report['datasets'][key]['trained_cnn']['test_accuracy']:.3f}", flush=True)
            del datasets
    return report
