# ============================================================
# PROJECT 4 — COMPLETE VALIDATION REPORT
# ============================================================
# Intern      : [Write Your Full Name Here]
# Organization: DecodeLabs
# Batch       : 2026
# File        : validation_report.py
# ============================================================
# PURPOSE     : Verify all 4 validation rules for both
#               Path 1 (OCR) and Path 2 (Object Detection)
# ============================================================

import cv2
import pytesseract
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

# --- Tesseract path ---
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

# --- Folder paths ---
IMAGES_DIR  = 'images'
OUTPUTS_DIR = 'outputs'
MODELS_DIR  = 'models'

print("=" * 60)
print("  PROJECT 4 — COMPLETE VALIDATION REPORT")
print("  Checking all 4 rules for both paths")
print("  Powered by: DecodeLabs | Batch 2026")
print("=" * 60)
print()

# ============================================================
# VALIDATION TRACKER
# ============================================================

# Store results for all 8 checks
# (4 rules × 2 paths)
validation_results = {
    'ocr': {
        'rule1': {'name': 'Library Integration',
                  'passed': False, 'detail': ''},
        'rule2': {'name': 'Pre-Processing Integrity',
                  'passed': False, 'detail': ''},
        'rule3': {'name': 'Confidence ≥ 80%',
                  'passed': False, 'detail': ''},
        'rule4': {'name': 'Visual Confirmation',
                  'passed': False, 'detail': ''}
    },
    'detection': {
        'rule1': {'name': 'Library Integration',
                  'passed': False, 'detail': ''},
        'rule2': {'name': 'Pre-Processing Integrity',
                  'passed': False, 'detail': ''},
        'rule3': {'name': 'Confidence ≥ 80%',
                  'passed': False, 'detail': ''},
        'rule4': {'name': 'Visual Confirmation',
                  'passed': False, 'detail': ''}
    }
}

# ============================================================
# PATH 1 VALIDATION — OCR
# ============================================================

print("━" * 60)
print("  VALIDATING PATH 1: OCR Text Recognition")
print("━" * 60)
print()

# ── RULE 1: Library Integration ──────────────────────────
print("  Checking Rule 1 — Library Integration...")
try:
    import pytesseract
    import cv2
    version = pytesseract.get_tesseract_version()
    cv_ver  = cv2.__version__

    validation_results['ocr']['rule1']['passed'] = True
    validation_results['ocr']['rule1']['detail'] = (
        f"pytesseract v{version} + "
        f"OpenCV v{cv_ver}"
    )
    print(f"  ✅ PASSED — pytesseract {version}"
          f" | OpenCV {cv_ver}")

except Exception as e:
    validation_results['ocr']['rule1']['detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ── RULE 2: Pre-Processing Integrity ─────────────────────
print("  Checking Rule 2 — Pre-Processing Integrity...")
try:
    # Load test image
    ocr_img_path = os.path.join(
        IMAGES_DIR, 'test_ocr.png')

    if not os.path.exists(ocr_img_path):
        raise FileNotFoundError(
            f"test_ocr.png not found in {IMAGES_DIR}/")

    img = cv2.imread(ocr_img_path)

    # Apply full pipeline
    gray      = cv2.cvtColor(img,
                              cv2.COLOR_BGR2GRAY)
    blurred   = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Verify each stage
    assert len(gray.shape) == 2, \
        "Grayscale failed"
    assert gray.shape == blurred.shape, \
        "Blur failed"
    assert np.unique(threshold).tolist() == [0, 255], \
        "Threshold failed"

    detail = (f"Grayscale {gray.shape} → "
              f"Blur (5x5) → "
              f"Threshold {np.unique(threshold)}")
    validation_results['ocr']['rule2']['passed'] = True
    validation_results['ocr']['rule2']['detail'] = detail

    print(f"  ✅ PASSED — {detail}")

except Exception as e:
    validation_results['ocr']['rule2']['detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ── RULE 3: Confidence ≥ 80% ─────────────────────────────
print("  Checking Rule 3 — Confidence Score ≥ 80%...")
try:
    # Run OCR with confidence data
    ocr_data = pytesseract.image_to_data(
        threshold,
        config='--psm 6',
        output_type=pytesseract.Output.DICT
    )

    # Get valid confidences
    confidences = [
        int(c) for c, t in
        zip(ocr_data['conf'], ocr_data['text'])
        if t.strip() and int(c) > 0
    ]

    if confidences:
        avg_conf = np.mean(confidences)
        passed   = avg_conf >= 80

        validation_results['ocr']['rule3']['passed'] = (
            passed)
        validation_results['ocr']['rule3']['detail'] = (
            f"Avg: {avg_conf:.1f}% | "
            f"Words: {len(confidences)}"
        )

        status = "✅ PASSED" if passed else "⚠️  REVIEW"
        print(f"  {status} — Average: {avg_conf:.1f}%"
              f" ({len(confidences)} words)")
    else:
        print("  ⚠️  No confident detections found")

except Exception as e:
    validation_results['ocr']['rule3']['detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ── RULE 4: Visual Confirmation ──────────────────────────
print("  Checking Rule 4 — Visual Confirmation...")
try:
    output_path = os.path.join(
        OUTPUTS_DIR, 'ocr_result.png')

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        validation_results['ocr']['rule4']['passed'] = True
        validation_results['ocr']['rule4']['detail'] = (
            f"outputs/ocr_result.png "
            f"({size/1024:.1f} KB)"
        )
        print(f"  ✅ PASSED — "
              f"outputs/ocr_result.png "
              f"({size/1024:.1f} KB)")
    else:
        print(f"  ❌ FAILED — "
              f"ocr_result.png not found in outputs/")
        print(f"     Run ocr_reader.py first!")

except Exception as e:
    validation_results['ocr']['rule4']['detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ============================================================
# PATH 2 VALIDATION — OBJECT DETECTION
# ============================================================

print("━" * 60)
print("  VALIDATING PATH 2: Object Detection")
print("━" * 60)
print()

# ── RULE 1: Library Integration ──────────────────────────
print("  Checking Rule 1 — Library Integration...")
try:
    # Try loading the model
    proto_path = os.path.join(
        MODELS_DIR, 'MobileNetSSD_deploy.prototxt')
    model_path = os.path.join(
        MODELS_DIR, 'MobileNetSSD_deploy.caffemodel')

    if not os.path.exists(proto_path):
        raise FileNotFoundError(
            "MobileNetSSD_deploy.prototxt not found")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "MobileNetSSD_deploy.caffemodel not found")

    net = cv2.dnn.readNetFromCaffe(
        proto_path, model_path)

    model_size = os.path.getsize(model_path)
    detail = (f"cv2.dnn loaded | "
              f"Model: {model_size/1024/1024:.1f}MB | "
              f"OpenCV {cv2.__version__}")

    validation_results['detection']['rule1'][
        'passed'] = True
    validation_results['detection']['rule1'][
        'detail'] = detail

    print(f"  ✅ PASSED — {detail}")

except Exception as e:
    validation_results['detection']['rule1'][
        'detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ── RULE 2: Pre-Processing Integrity ─────────────────────
print("  Checking Rule 2 — Pre-Processing Integrity...")
try:
    detect_img_path = os.path.join(
        IMAGES_DIR, 'my_photo.jpg')

    if not os.path.exists(detect_img_path):
        raise FileNotFoundError(
            "my_photo.jpg not found")

    img   = cv2.imread(detect_img_path)
    (h,w) = img.shape[:2]

    # Create blob — the required pre-processing step
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    # Verify blob shape
    assert blob.shape == (1, 3, 300, 300), \
        f"Unexpected blob shape: {blob.shape}"

    detail = (f"Blob: {blob.shape} | "
              f"Scale: 0.007843 | "
              f"Mean: 127.5")
    validation_results['detection']['rule2'][
        'passed'] = True
    validation_results['detection']['rule2'][
        'detail'] = detail

    print(f"  ✅ PASSED — {detail}")

except Exception as e:
    validation_results['detection']['rule2'][
        'detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ── RULE 3: Confidence ≥ 80% ─────────────────────────────
print("  Checking Rule 3 — Confidence Score ≥ 80%...")
try:
    # Run detection
    net.setInput(blob)
    detections = net.forward()

    CLASSES = [
        "background", "aeroplane", "bicycle",
        "bird",       "boat",      "bottle",
        "bus",        "car",       "cat",
        "chair",      "cow",       "diningtable",
        "dog",        "horse",     "motorbike",
        "person",     "pottedplant","sheep",
        "sofa",       "train",     "tvmonitor"
    ]

    # Get all confident detections
    conf_scores = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf >= 0.5:
            conf_scores.append(conf * 100)

    if conf_scores:
        avg_conf = np.mean(conf_scores)
        max_conf = max(conf_scores)
        passed   = max_conf >= 80

        validation_results['detection']['rule3'][
            'passed'] = passed
        validation_results['detection']['rule3'][
            'detail'] = (
            f"Best: {max_conf:.1f}% | "
            f"Detections: {len(conf_scores)}"
        )

        status = "✅ PASSED" if passed else "⚠️  REVIEW"
        print(f"  {status} — Best: {max_conf:.1f}%"
              f" ({len(conf_scores)} detections)")
    else:
        print("  ⚠️  No detections above 50% threshold")
        print("     Try with an image containing clear"
              " objects")
        validation_results['detection']['rule3'][
            'detail'] = "No confident detections"

except Exception as e:
    validation_results['detection']['rule3'][
        'detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ── RULE 4: Visual Confirmation ──────────────────────────
print("  Checking Rule 4 — Visual Confirmation...")
try:
    output_path = os.path.join(
        OUTPUTS_DIR, 'detection_result.png')

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        validation_results['detection']['rule4'][
            'passed'] = True
        validation_results['detection']['rule4'][
            'detail'] = (
            f"outputs/detection_result.png "
            f"({size/1024:.1f} KB)"
        )
        print(f"  ✅ PASSED — "
              f"outputs/detection_result.png "
              f"({size/1024:.1f} KB)")
    else:
        print(f"  ❌ FAILED — "
              f"detection_result.png not found")
        print(f"     Run object_detector.py first!")

except Exception as e:
    validation_results['detection']['rule4'][
        'detail'] = str(e)
    print(f"  ❌ FAILED — {e}")

print()

# ============================================================
# FINAL VALIDATION SUMMARY
# ============================================================

print("=" * 60)
print("  COMPLETE VALIDATION SUMMARY")
print("=" * 60)
print()

# Count passes
ocr_passes = sum(
    1 for r in validation_results['ocr'].values()
    if r['passed'])
det_passes = sum(
    1 for r in validation_results['detection'].values()
    if r['passed'])
total_passes = ocr_passes + det_passes

# Print table
print(f"  {'Rule':<32} {'OCR':^8} {'Detection':^10}")
print(f"  {'─'*55}")

rule_names = [
    "Rule 1 — Library Integration",
    "Rule 2 — Pre-Processing Integrity",
    "Rule 3 — Confidence ≥ 80%",
    "Rule 4 — Visual Confirmation"
]

for rule_key, rule_name in zip(
        ['rule1','rule2','rule3','rule4'],
        rule_names):

    ocr_pass = validation_results['ocr'][
        rule_key]['passed']
    det_pass = validation_results['detection'][
        rule_key]['passed']

    ocr_sym = "  ✅" if ocr_pass else "  ❌"
    det_sym = "    ✅  " if det_pass else "    ❌  "

    print(f"  {rule_name:<32}"
          f"{ocr_sym:^8}{det_sym:^10}")

print(f"  {'─'*55}")
print(f"  {'TOTAL PASSED':<32}"
      f"  {ocr_passes}/4    {det_passes}/4")
print()

# Overall status
if total_passes == 8:
    print("  🎉 PERFECT SCORE! ALL 8 CHECKS PASSED!")
    print("     Both paths fully validated!")
elif total_passes >= 6:
    print(f"  ✅ GOOD: {total_passes}/8 checks passed")
    print("     Review any ❌ items above")
else:
    print(f"  ⚠️  {total_passes}/8 checks passed")
    print("     Run both path scripts first")

print()
print("=" * 60)

# ── GENERATE VALIDATION CHART ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rules_short = [
    'Library\nIntegration',
    'Pre-Processing\nIntegrity',
    'Confidence\n≥ 80%',
    'Visual\nConfirmation'
]

# Colors for pass/fail
def get_colors(path_name):
    return ['#2ecc71' if
            validation_results[path_name][f'rule{i}']
            ['passed']
            else '#e74c3c'
            for i in range(1, 5)]

# OCR chart
ocr_colors = get_colors('ocr')
bars1 = axes[0].bar(rules_short,
                    [1, 1, 1, 1],
                    color=ocr_colors,
                    edgecolor='white',
                    linewidth=2,
                    width=0.5)

for bar, (key, result) in zip(
        bars1,
        validation_results['ocr'].items()):
    status = "✅ PASS" if result['passed'] else "❌ FAIL"
    axes[0].text(
        bar.get_x() + bar.get_width()/2,
        0.5,
        status,
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color='white'
    )

axes[0].set_title(
    f'Path 1: OCR Validation\n'
    f'{ocr_passes}/4 Rules Passed',
    fontweight='bold', fontsize=12)
axes[0].set_ylim(0, 1.3)
axes[0].set_yticks([])
axes[0].set_xlabel('Validation Rules',
                    fontsize=10)

# Detection chart
det_colors = get_colors('detection')
bars2 = axes[1].bar(rules_short,
                    [1, 1, 1, 1],
                    color=det_colors,
                    edgecolor='white',
                    linewidth=2,
                    width=0.5)

for bar, (key, result) in zip(
        bars2,
        validation_results['detection'].items()):
    status = "✅ PASS" if result['passed'] else "❌ FAIL"
    axes[1].text(
        bar.get_x() + bar.get_width()/2,
        0.5,
        status,
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color='white'
    )

axes[1].set_title(
    f'Path 2: Detection Validation\n'
    f'{det_passes}/4 Rules Passed',
    fontweight='bold', fontsize=12)
axes[1].set_ylim(0, 1.3)
axes[1].set_yticks([])
axes[1].set_xlabel('Validation Rules',
                    fontsize=10)

plt.suptitle(
    f'Project 4 — Complete Validation Report\n'
    f'Total: {total_passes}/8 Checks Passed | '
    f'DecodeLabs Batch 2026',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()

chart_path = os.path.join(
    OUTPUTS_DIR, 'validation_report.png')
plt.savefig(chart_path,
            dpi=150, bbox_inches='tight')
plt.show()

print(f"  Validation chart saved: {chart_path}")
print()