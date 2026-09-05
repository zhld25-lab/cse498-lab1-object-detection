# CSE398/498 Lab 1 — Object Detection with YOLO11 and YOLOE

Zhenzhe Luo (zhld25) · Lehigh University · Fall 2026

Three experiments comparing pretrained YOLO11, fine-tuned YOLO11, and
open-vocabulary YOLOE on the same set of real-world objects.

## Test objects

Three items were chosen because they are absent from or poorly covered by the
COCO label set:

| Class | Object | Why it was chosen |
|---|---|---|
| chip_bag | Ruffles cheddar & sour cream bag | No COCO class for snack packaging; deformable shape |
| pepper_shaker | Pepper shaker bottle | COCO has `bottle`, but not the specific product |
| earphone_case | Wireless earbuds charging case | Shape resembles COCO `mouse` |

## Results summary

Same 17 test images (captured with an external webcam) across all methods,
`conf=0.25` unless noted.

| Method | Correct | Training data | Class flexibility |
|---|---|---|---|
| Pretrained YOLO11n | 0 / 17 | none | fixed 80 COCO classes |
| Fine-tuned YOLO11n | 17 / 17 | 47 images + labels | fixed 3 classes |
| YOLOE, prompt set A | 3 / 17 | none | any text at inference |
| YOLOE, prompt set B (conf=0.05) | 17 / 17 | none | any text at inference |

Prompt set A used the class names above; set B used appearance-based
descriptions (snack package, spice bottle, small white plastic case). Full
analysis is in `report/Lab1_Report.pdf`.

## Repository layout

    .
    ├── main.py                              original webcam demo (unmodified)
    ├── models/yolo11n.pt                    pretrained weights
    ├── experiment1_yolo11_baseline/
    │   ├── capture_baseline.py              capture + log tool
    │   ├── fill_ground_truth.py             annotates the CSV
    │   ├── list_cams.py                     camera index helper
    │   └── results/
    │       ├── raw/                         17 unannotated test images
    │       ├── annotated/                   detections drawn
    │       └── detections.csv               per-detection log
    ├── experiment2_yolo11_finetuning/
    │   └── results/
    │       ├── best.pt                      fine-tuned weights
    │       ├── test01-17.jpg                predictions on the test images
    │       ├── labels/                      YOLO-format outputs with conf
    │       ├── confusion_matrix.png
    │       └── results.png                  training curves
    ├── experiment3_yoloe/
    │   ├── yoloe_text/                      prompt set A, conf=0.25
    │   ├── yoloe_text_conf005/              prompt set A, conf=0.05
    │   └── yoloe_text_alt/                  prompt set B, conf=0.05
    ├── dataset/raw_train/                   47 training images
    └── report/Lab1_Report.pdf

## Setup

Requires Python 3.10+ and a webcam for Experiment 1.

    git clone https://github.com/zhld25-lab/cse498-lab1-object-detection.git
    cd cse498-lab1-object-detection
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install ultralytics opencv-python

On macOS or Linux use `source .venv/bin/activate` instead.

No GPU is needed to reproduce the inference results; training was done on a
Colab T4.

## Experiment 1 — Pretrained YOLO11 baseline

The original webcam demo, unchanged:

    python main.py

To reproduce the logged baseline, `capture_baseline.py` runs the same model at
`conf=0.25` and writes each detection to CSV. Press `s` to capture, `[` and `]`
to adjust the threshold, `q` to quit.

    python experiment1_yolo11_baseline\capture_baseline.py

The 17 images already captured are in `experiment1_yolo11_baseline\results`.

Pretrained YOLO11n misclassified every one of them: the chip bag came back as
`person` and `suitcase`, the earphone case as `mouse` (0.97 confidence on
test12), and the pepper shaker only ever as the generic `bottle`.

## Experiment 2 — Fine-tuned YOLO11

Dataset: 47 phone photos, annotated in Roboflow, split 33/9/5, augmented 3x to
112 images. Trained 100 epochs from `yolo11n.pt` on a Colab T4, about five
minutes.

Reproduce the predictions with the saved weights:

    yolo task=detect mode=predict model=experiment2_yolo11_finetuning\results\best.pt source=experiment1_yolo11_baseline\results\raw conf=0.25 save=True

Validation set (9 images): mAP50 = 0.995, mAP50-95 = 0.749. These numbers are
optimistic given the small validation split. The meaningful test is the 17
webcam images, which the model had never seen and which came from a different
camera than the training photos.

Training notebook: train-yolo11-object-detection-on-custom-dataset.ipynb
(Roboflow), with the dataset download cell pointed at this project.

## Experiment 3 — YOLOE open-vocabulary detection

Run in Colab with `yoloe-11l-seg.pt`:

    from ultralytics import YOLOE

    model = YOLOE("yoloe-11l-seg.pt")
    names = ["chip bag", "pepper shaker", "earphone case"]
    model.set_classes(names, model.get_text_pe(names))
    model.predict(source="test_images/", conf=0.25, save=True)

Notebook: zero-shot-object-detection-and-segmentation-with-yoloe.ipynb
(Roboflow).

The same model on the same images detected 3/17 with the class names above and
17/17 with appearance-based prompts. Notably "wireless earbuds case" never
matched while "small white plastic case" matched all seven, which suggests
YOLOE aligns text against visual features rather than functional identity.

## Notes

- All three methods were evaluated on the identical 17 images at matching
  confidence thresholds, so the comparison isolates the method rather than the
  data.
- Training images were taken with a phone; test images with an external webcam.
  The fine-tuned model transferred across that gap, which suggests it learned
  object features rather than capture conditions.
- Roboflow API keys are not committed. Insert your own key in the dataset
  download cell before running the training notebook.
