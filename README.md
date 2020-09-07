# Super 8 IVTC
This is a work-in-progress. Parts of it might be useful, but its intended use is in a Python shell, not as a standalone app,
and it's only been tuned for a single dataset.

This is **experimental** code for performing [IVTC](https://en.wikipedia.org/wiki/Telecine)
on content that was projected, then recorded with a camera. It was originally
written to estimate the frame rate to use in Avisynth's [TDecimate's](http://avisynth.nl/index.php/TIVTC/TDecimate)
mode 2, but it can also do the decimation on its own.

TDecmate has two modes (mode 0 and mode 2) that are useful for IVTC of Super 8 content, but each
has a problem with real-world captures where the precise framerate of the projector
isn't known. In mode 0, if the projector doesn't run consistently at precisely
18 fps, the M-in-N numbers will get silly-large to accommodate odd framerates. In mode 2,
this isn't an issue, but knowing the framerate is.

This code minimizes a loss function that targets a known framerate and penalizes
differences between frames. In addition to penalizing by frame delta, you can add known
ground truth sequences to strongly hint at the right frames to drop and for measuring the
impact of changes to the model.

Without modifications, this implementation only works for content captured at 29.97 fps that was
originally 15-20 fps. The implementation assumes every video can be represented as a sequence of
either keep-keep-drop or keep-drop, and even then, the penalties haven't been tuned to be flexible.
That said, the algorithm and technique could be extended; the heuristics for the cost function just
need to be improved.

## Preparing video

This script works with pngs in a directory:

```shell script
mkdir src_frames
ffmpeg -i input.mkv src_frames/out%06d.png
```
 
## Ground truth file

The ground truth file is a yaml file. The first frame in a series of duplicates should be marked with
a `0`, e.g. for `ABBCDEEF`, you'd mark `1: [0, 1, 1, 0, 1, 1]`. Sequences can be of any length, but
they can't overlap. Validation is split by sequence.

This file is only needed for validation, but it exists because it was easier to add known dropped and
retained frames than tuning the frame delta function.

```yaml
1000: [0, 1, 2]
1002: [0, 1, 2]
```

## Evaluate model

```python
import super8

ground_truth = super8.load_ground_truth("ground_truth.yaml")
src_files, diffs, std_devs = super8.compute_diffs("src_frames")
accuracy = super8.eval_model(diffs, ground_truth, super8.CostFunctionParams())

# Make additional calls to eval_model, tuning params
```

## Run

```python
import super8

ground_truth = super8.load_ground_truth("ground_truth.yaml")
src_files, diffs, std_devs = super8.compute_diffs("src_frames")
super8.run(src_files, "dest_frames", diffs, std_devs, ground_truth, super8.CostFunctionParams())
```
