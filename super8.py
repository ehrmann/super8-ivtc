import functools
import math
import os
import random
import shutil
import statistics
from typing import Mapping, Callable, MutableMapping, NamedTuple, Sequence, Optional, MutableSequence, Tuple

import imageio
import numpy as np
import yaml
from tqdm import tqdm


class Cost(NamedTuple):
    cost: int
    sequence: Sequence[bool]


class CostFunctionParams(NamedTuple):
    recorded_fps: float = 29.97
    target_fps: float = 18
    f2: float = 1.0
    f3: float = 1.0


Memos = MutableMapping[int, Cost]
GroundTruth = Mapping[int, Sequence[bool]]
FlatGroundTruth = Mapping[int, bool]
CostFunction = Callable[[], float]


def load_ground_truth(filename) -> GroundTruth:
    with open(filename) as f:
        return {
            k: [bool(u) for u in v]
            for k, v in
            yaml.load(f, Loader=yaml.FullLoader).items()
        }


def compute_diffs(src_dir) -> Tuple[Sequence[str], Sequence[float], Sequence[float]]:
    files = os.listdir(src_dir)
    files = sorted(f for f in files if f.endswith(".png"))

    src_files = []
    diffs = []
    std_devs = []

    previous = None

    for file in tqdm(files):
        file = src_dir + os.path.sep + file
        src_files.append(file)
        image = np.array(imageio.imread(file))

        std_devs.append(float(np.std(image)))
        image_min = np.min(image)
        image_max = np.max(image)

        if image_min != image_max:
            image_scale = 256 / (image_max - image_min)
            image -= image_min
            image = image * image_scale
        else:
            image *= 0

        if previous is not None:
            diff = image - previous
            diffs.append(float(np.sum(np.abs(diff) ** 2)))

        previous = image

    return src_files, diffs, std_devs


def _est_fps(i: int, memos: Memos, window: int, recorded_fps: float) -> Optional[float]:
    total_frames = 0
    included_frames = 0
    j = i
    while j > 0 and i - j < window:
        seq = memos[j][1]
        total_frames += len(seq)
        included_frames += sum(seq)
        j -= len(seq)

    return (included_frames / total_frames) * recorded_fps if total_frames > 0 else None


def _cost_function(
        memos: Memos,
        flat_ground_truth: FlatGroundTruth,
        diffs: Sequence[float],
        mean_diff: float,
        i: int,
        params: CostFunctionParams = CostFunctionParams()
) -> Cost:
    discard = diffs[i - 1]
    penalty = len(diffs) * mean_diff
    est_fps = functools.partial(_est_fps, memos=memos, window=15, recorded_fps=params.recorded_fps)

    vals: MutableSequence[Cost] = []
    if i >= 3:
        # This worked better than biasing using fps
        factor = .5

        """
        x = est_fps(i - 3)
        if x:
            if x < target_fps:
                factor *= .5
            elif x > target_fps:
                factor *= 2
        """

        vals.append(Cost(memos[i - 3][0] + factor * mean_diff + params.f3 * discard, [True, True, False]))
    if i >= 2:
        x = est_fps(i - 2)
        factor = 1
        if x:
            if x < params.target_fps:
                factor *= 2
            elif x > params.target_fps:
                factor *= .5

        vals.append(Cost(memos[i - 2][0] + factor * mean_diff + params.f2 * discard, [True, False]))
    if i == 1:
        vals.append(Cost(memos[i - 1][0] + discard, [False]))

    if i == len(diffs):
        vals.extend((
            Cost(memos[i - 1][0], [True]),
            Cost(memos[i - 2][0], [True, True]),
        )
        )

    # Allow single frames, but make them expensive
    vals.append(Cost(memos[i - 1][0] + discard * 40, [False]))
    vals.append(Cost(memos[i - 1][0] + discard * 40, [True]))
    for val in vals:
        for j in range(-len(val[1]), 0):
            if flat_ground_truth.get(i + j, val[1][j]) != val[1][j]:
                val[0] += penalty

    return min(vals, key=lambda cost: cost.cost)


def _flatten_ground_truth(ground_truth: GroundTruth) -> FlatGroundTruth:
    result = {}
    for base, values in ground_truth.items():
        for offset, value in enumerate(values):
            if (base + offset) in result:
                raise Exception(f"{base + offset} already set")
            result[base + offset] = value
    return result


def _refine_includes(include_seq: Sequence[bool], std_devs: Sequence[float]) -> Sequence[bool]:
    if len(include_seq) != len(std_devs):
        raise ValueError()

    refined = list(include_seq)
    i = 1
    while i < len(refined):
        if not refined[i] and refined[i - 1] and std_devs[i] > std_devs[i - 1]:
            refined[i] = 1
            refined[i - 1] = 0

            i += 1
        i += 1

    return refined


def _copy_files(src_files: Sequence[str], dest_dir: str, include_seq: Sequence[bool]) -> None:
    if len(src_files) != len(include_seq):
        raise ValueError()

    padding = math.ceil(math.log10(len(include_seq)))
    filename = f"out%0{padding}d.png"
    i = 0
    j = 0
    while i < len(include_seq):
        if include_seq[i]:
            shutil.copyfile(
                src_files[i],
                dest_dir + os.path.sep + filename % (j,)
            )
            j += 1
        i += 1


def _compute(
        diffs: Sequence[float],
        ground_truth: GroundTruth,
        cost_function_params: CostFunctionParams
) -> Sequence[bool]:
    memos = {}

    flat_ground_truth = _flatten_ground_truth(ground_truth)
    mean_diff = statistics.mean(diffs)

    for i in range(0, len(diffs) + 1):
        memos[i] = _cost_function(memos, flat_ground_truth, diffs, mean_diff, i, cost_function_params)

    result = []
    i = len(diffs)
    while i > 0:
        result.extend(reversed(memos[i].sequence))
        i -= len(memos[i][1])

    result.append(True)
    result.reverse()
    return result


def eval_model(diffs: Sequence[float], ground_truth: GroundTruth, cost_function_params: CostFunctionParams) -> float:
    _flatten_ground_truth(ground_truth)
    gt_keys = list(ground_truth)

    for _ in tqdm(range(0, 20)):
        random.shuffle(gt_keys)
        test_gt = {k: ground_truth[k] for k in gt_keys[0: len(gt_keys) * 8 // 10]}
        validate_gt = _flatten_ground_truth(
            {k: ground_truth[k] for k in gt_keys[len(gt_keys) * 8 // 10:]}
        )

        include_seq = _compute(diffs, test_gt, cost_function_params)

        # print(f"fps = {cost_function_params.recorded_fps * sum(include_seq) / len(include_seq)}")

        match = 0
        for k, v in validate_gt.items():
            if include_seq[k] == v:
                match += 1

        return match / len(validate_gt)


def run(
        src_files: Sequence[str],
        dest_dir: str,
        diffs: Sequence[float],
        std_devs: Sequence[float],
        ground_truth: GroundTruth,
        cost_function_params: CostFunctionParams
) -> None:
    include_seq = _compute(diffs, ground_truth, cost_function_params)
    include_seq = _refine_includes(include_seq, std_devs)
    _copy_files(src_files, dest_dir, include_seq)
