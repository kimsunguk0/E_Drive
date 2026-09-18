"""Read-only, CPU analysis of frozen A2 BASE/MH4 predictions and provided commands."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/NHNHOME/data/sukim/adcl")
OUT = ROOT / "reports/md_a2_turn_analysis_20260918"
WEIGHTS = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36


def groups_for(pred, gt, mask, sessions, scenarios):
    err = pred - gt
    distance = np.linalg.norm(err, axis=-1)
    score = distance @ WEIGHTS
    delta = np.diff(np.concatenate([np.zeros((len(gt), 1, 2)), gt], 1), axis=1)
    length = np.linalg.norm(delta, axis=-1)
    tangent = delta / np.maximum(length[..., None], 1e-8)
    valid = length > .05
    weight = valid[mask] * WEIGHTS
    long = np.abs((err * tangent).sum(-1))
    lat = np.abs(err[..., 0] * tangent[..., 1] - err[..., 1] * tangent[..., 0])
    n = int(mask.sum())
    if not n:
        return {"n": 0}
    pred_class = np.where(pred[:, -1, 1] >= 2, 1, np.where(pred[:, -1, 1] <= -2, 0, 2))
    return dict(n=n, fraction=n / len(gt), sessions=len(set(sessions[mask])),
        scenarios=len(set(scenarios[mask])), PREFIX=float(score[mask].mean()),
        total_score_contribution=float(score[mask].sum() / len(gt)),
        fraction_of_total_score=float(score[mask].sum() / score.sum()),
        prefix_1s=float(distance[mask, :2].mean()),
        prefix_2s=float(distance[mask, :4].mean()),
        prefix_3s=float(distance[mask].mean()),
        first2s_weighted_contribution_in_group=float((distance[mask, :4] * WEIGHTS[:4]).sum(1).mean()),
        last2points_weighted_contribution_in_group=float((distance[mask, 4:] * WEIGHTS[4:]).sum(1).mean()),
        weighted_abs_long_m=float((long[mask] * weight).sum() / weight.sum()),
        weighted_abs_lat_m=float((lat[mask] * weight).sum() / weight.sum()),
        projection_denominator="weighted valid GT tangents, length >0.05m; projections are not additive PREFIX",
        endpoint_abs_xy_error=np.abs(err[mask, -1]).mean(0).tolist(),
        endpoint_opposite_lateral_sign_n=int((mask & (gt[:, -1, 1] * pred[:, -1, 1] < 0)).sum()),
        predicted_geometry_counts={str(c): int((mask & (pred_class == c)).sum()) for c in [0, 1, 2]})


def main():
    s = np.load(OUT / "prediction_snapshot.npz", allow_pickle=False)
    metadata = np.load(OUT / "analysis_metadata.npz", allow_pickle=False)
    manifest = json.loads((OUT / "snapshot_manifest.json").read_text())
    rows = s["BASE_NOM_rows"]
    gt = s["BASE_NOM_gt"]
    cache = np.load("/tmp/pm97/data/etri/ego_cache.npz", allow_pickle=False)
    sessions = s["BASE_NOM_session"]
    scenarios = cache["scenarios"][cache["scen_idx"][rows]]
    assert np.array_equal(gt, cache["fut"][rows])
    assert np.array_equal(rows, metadata["rows"])
    geometry = np.where(gt[:, -1, 1] >= 2, 1, np.where(gt[:, -1, 1] <= -2, 0, 2))
    assert np.array_equal(geometry, metadata["vad_cmd"])
    masks = {"left_3s_lateral_ge_2m": geometry == 1,
             "right_3s_lateral_le_minus2m": geometry == 0,
             "other_3s": geometry == 2}
    semantic = {str(label): metadata["meta"] == i for i, label in enumerate(metadata["labels"])}
    result = dict(scope="Read-only frozen V0 analysis; no model or training change; 17130 matched comparison",
        official_weights=WEIGHTS.tolist(), n=len(rows),
        definitions={"geometry": "GT 3s endpoint lateral y >=2m left, <=-2m right. Includes bends/lane changes; not semantic intersection turns.",
                     "semantic": "Raw provided command labels, read via existing timestamp-mapped ego cache.",
                     "nonstop": "No stop in the bucket definition; includes turns and lane changes."},
        snapshot_manifest=manifest, models={})
    for arm in ["BASE_NOM", "MH4_NOM", "A2_REAL"]:
        assert np.array_equal(rows, s[arm + "_rows"])
        assert np.array_equal(gt, s[arm + "_gt"])
        pred = s[arm + "_pred"]
        result["models"][arm] = dict(
            PREFIX=float((np.linalg.norm(pred - gt, axis=-1) @ WEIGHTS).mean()),
            geometry={name: groups_for(pred, gt, mask, sessions, scenarios) for name, mask in masks.items()},
            semantic={name: groups_for(pred, gt, mask, sessions, scenarios) for name, mask in semantic.items()})
    current = result["models"]["MH4_NOM"]
    turn_contribution = sum(current["geometry"][n]["total_score_contribution"] for n in list(masks)[:2])
    semantic_turn_contribution = sum(current["semantic"][n]["total_score_contribution"] for n in ["TURN_LEFT", "TURN_RIGHT"])
    result["bounded_subgroup_scenarios"] = dict(
        perfect_202_geometry_rows_only=current["PREFIX"] - turn_contribution,
        halve_202_geometry_errors_only=current["PREFIX"] - .5 * turn_contribution,
        perfect_81_semantic_turn_rows_only=current["PREFIX"] - semantic_turn_contribution,
        turn_error_reduction_fraction_to_015_if_all_other_predictions_fixed=(current["PREFIX"] - .15) / turn_contribution,
        scope="Arithmetic scenarios, not oracle model performance or bounds on all possible command benefits.")
    goal = metadata["goal"]
    goal_class = np.where(goal[:, 1] >= 2, 1, np.where(goal[:, 1] <= -2, 0, 2))
    result["goal5s_vs_geometry3s"] = dict(
        class_order=["right", "left", "other"],
        matrix_rows_goal_columns_gt3s=[[int(((goal_class == a) & (geometry == b)).sum()) for b in [0, 1, 2]] for a in [0, 1, 2]],
        different_rows=int((goal_class != geometry).sum()),
        warning="Hard threshold illustration only. The actual model uses continuous goal; these are not model navigation errors.")
    model_pred = s["MH4_NOM_pred"]
    result["turn_shape"] = {}
    for name, mask in list(masks.items())[:2]:
        outward = (model_pred - gt)[mask, :, 1] * np.sign(gt[mask, -1, 1, None])
        under_threshold = mask & (np.abs(model_pred[:, -1, 1]) < 2)
        result["turn_shape"][name] = dict(
            signed_outward_lateral_error_by_time_m=outward.mean(0).tolist(),
            inside_2m_threshold_n=int(under_threshold.sum()),
            missed_threshold_GT_abs_y_quantiles=np.quantile(np.abs(gt[under_threshold, -1, 1]), [0, .5, 1]).tolist(),
            nonstop_n=int((mask & (s["BASE_NOM_bucket"] == "nonstop")).sum()),
            interpretation="Negative endpoint mean is less lateral displacement than GT; progress and curvature are not separated by this statistic.")

    test_files = sorted(Path("/tmp/etri_test").glob("*/command.parquet"))
    def read_command(path):
        d = pq.read_table(path).to_pydict()
        return str(d["command"][0]), int(np.argmax(d["vad_cmd"][0]))
    if test_files:
        with ThreadPoolExecutor(max_workers=8) as pool:
            commands = list(pool.map(read_command, test_files))
        result["test_provided_input_counts"] = dict(
            n=len(commands), command=dict(Counter(x[0] for x in commands)),
            vad_cmd=dict(Counter(x[1] for x in commands)),
            source="/tmp/etri_test/*/command.parquet; no hidden GT or prediction used")

    split = json.loads((ROOT / "data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json").read_text())["splits"]
    all_names = cache["scenarios"][cache["scen_idx"]]
    train_mask = np.isin(all_names, split["train"]) & (cache["frame"] >= 30)
    result["train_provided_command_counts"] = {
        str(label): int((train_mask & (cache["meta"] == i)).sum()) for i, label in enumerate(cache["meta_labels"])}
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / "turn_analysis.json").write_text(json.dumps(result, indent=2) + "\n")

    score = np.linalg.norm(model_pred - gt, axis=-1) @ WEIGHTS
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    examples = []
    for col, (label, mask) in enumerate([("Left", geometry == 1), ("Right", geometry == 0)]):
        order = np.flatnonzero(mask)[np.argsort(score[mask])]
        for row, (which, index) in enumerate([("median", order[len(order) // 2]), ("largest error", order[-1])]):
            ax = axes[row, col]
            for name, array, style, color in [
                ("GT", gt, "-", "black"), ("BASE", s["BASE_NOM_pred"], "--", "#2874a6"),
                ("MH4", model_pred, "-", "#d35400")]:
                xy = np.concatenate([np.zeros((1, 2)), array[index]])
                ax.plot(-xy[:, 1], xy[:, 0], style, color=color, marker="o", markersize=3, label=name)
            ax.scatter([0], [0], marker="^", c="green", s=40)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=.25)
            ax.set_xlabel("Lateral coordinate (right positive, m)")
            ax.set_ylabel("Forward coordinate (m)")
            ax.set_title(f"{label}, {which}: MH4 PREFIX {score[index]:.3f}\nrow {rows[index]}, frame {s['BASE_NOM_frame'][index]}")
            ax.legend(fontsize=8)
            examples.append(dict(direction=label, selection=which, row=int(rows[index]),
                frame=int(s["BASE_NOM_frame"][index]), session=str(sessions[index]),
                scenario=str(scenarios[index]), meta_command=str(metadata["labels"][metadata["meta"][index]]),
                goal5s_xy=goal[index].tolist(), gt=gt[index].tolist(),
                base=s["BASE_NOM_pred"][index].tolist(), mh4=model_pred[index].tolist(),
                mh4_PREFIX=float(score[index])))
    fig.suptitle("Actual 3-second trajectories: matched BASE vs MH4 at update 17,130\nGT lateral threshold groups; median and maximum error within each group", fontsize=12)
    fig.savefig(OUT / "turn_examples.png", dpi=170)
    fig.savefig(OUT / "turn_examples.pdf")
    plt.close(fig)
    (OUT / "turn_examples.json").write_text(json.dumps(examples, indent=2) + "\n")

    semantic_examples = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for col, (label, mask) in enumerate([
        ("TURN_LEFT", (metadata["meta"] == 1) & (geometry == 1)),
        ("TURN_RIGHT", (metadata["meta"] == 2) & (geometry == 0))]):
        order = np.flatnonzero(mask)[np.argsort(score[mask])]
        for row, (which, index) in enumerate([("median", order[len(order) // 2]), ("largest error", order[-1])]):
            ax = axes[row, col]
            for name, array, style, color in [
                ("GT", gt, "-", "black"), ("BASE", s["BASE_NOM_pred"], "--", "#2874a6"),
                ("MH4", model_pred, "-", "#d35400")]:
                xy = np.concatenate([np.zeros((1, 2)), array[index]])
                ax.plot(-xy[:, 1], xy[:, 0], style, color=color, marker="o", markersize=3, label=name)
            ax.scatter([0], [0], marker="^", c="green", s=40)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=.25)
            ax.set_xlabel("Lateral coordinate (right positive, m)")
            ax.set_ylabel("Forward coordinate (m)")
            ax.set_title(f"{label}, {which}: MH4 PREFIX {score[index]:.3f}\nrow {rows[index]}, frame {s['BASE_NOM_frame'][index]}")
            ax.legend(fontsize=8)
            semantic_examples.append(dict(direction=label, selection=which, subgroup_n=len(order),
                row=int(rows[index]), frame=int(s["BASE_NOM_frame"][index]),
                session=str(sessions[index]), scenario=str(scenarios[index]),
                gt=gt[index].tolist(), base=s["BASE_NOM_pred"][index].tolist(),
                mh4=model_pred[index].tolist(), mh4_PREFIX=float(score[index])))
    fig.suptitle("Provided TURN_LEFT / TURN_RIGHT with |GT 3s lateral| >= 2m\nMatched BASE vs MH4 at update 17,130; median and maximum error", fontsize=12)
    fig.savefig(OUT / "semantic_turn_examples.png", dpi=170)
    fig.savefig(OUT / "semantic_turn_examples.pdf")
    plt.close(fig)
    (OUT / "semantic_turn_examples.json").write_text(json.dumps(semantic_examples, indent=2) + "\n")
    print(json.dumps({"geometry": current["geometry"], "bounded_subgroup_scenarios": result["bounded_subgroup_scenarios"],
        "test_provided_input_counts": result.get("test_provided_input_counts"), "examples": [{k: x[k] for k in ["direction", "selection", "row", "meta_command", "mh4_PREFIX"]} for x in examples]}, indent=2))

if __name__ == "__main__":
    main()
