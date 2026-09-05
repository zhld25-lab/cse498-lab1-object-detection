import csv

CSV_PATH = "results/exp1_baseline/detections.csv"

# 按图片编号区间指定真实物体
RANGES = [
    (1, 5, "chip_bag"),
    (6, 10, "pepper_shaker"),
    (11, 17, "earphone_case"),
]

# 预测类别命中哪个真实物体时算「部分正确」（超类正确但粒度不够）
PARTIAL = {"pepper_shaker": {"bottle"}}


def truth_for(image_id):
    n = int(image_id.replace("test", ""))
    for lo, hi, name in RANGES:
        if lo <= n <= hi:
            return name
    return ""


with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    gt = truth_for(r["image_id"])
    r["ground_truth"] = gt
    pred = r["predicted_class"]
    if pred == "NO_DETECTION":
        r["correct"] = "miss"
    elif pred in PARTIAL.get(gt, set()):
        r["correct"] = "partial"
    else:
        r["correct"] = "wrong"

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"updated {len(rows)} rows")