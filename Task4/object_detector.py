# ============================================================
# PROJECT 4 — PATH 2: Object Detection
# ============================================================
# Intern      : [Write Your Full Name Here]
# Organization: DecodeLabs
# Batch       : 2026
# File        : object_detector.py
# ============================================================
# OBJECTIVE   : Detect objects in images using MobileNet-SSD
# PIPELINE    : Load Model → Pre-process → Detect → Draw
# ALGORITHM   : Single Shot Detector (SSD) + MobileNet
# ============================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ⚙️  USER CONFIGURATION — CHANGE THIS LINE ONLY
# ============================================================

# 👇 Put the filename of YOUR photo here (must be in images/)
MY_IMAGE_FILENAME = "my_photo1.jpg"

# ============================================================

# --- Folder paths ---
IMAGES_DIR  = 'images'
OUTPUTS_DIR = 'outputs'
MODELS_DIR  = 'models'

os.makedirs(IMAGES_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

print("=" * 60)
print("  PROJECT 4 — PATH 2: Object Detection")
print("  Model     : MobileNet-SSD (Pre-trained)")
print("  Dataset   : VOC (20 object classes)")
print("  Powered by: DecodeLabs | Batch 2026")
print("=" * 60)
print()

# ============================================================
# STEP 1: DEFINE CLASS LABELS
# ============================================================

print("[ STEP 1 ] Loading Class Labels...")
print("-" * 60)

CLASSES = [
    "background", "aeroplane",  "bicycle",    "bird",
    "boat",       "bottle",     "bus",         "car",
    "cat",        "chair",      "cow",         "diningtable",
    "dog",        "horse",      "motorbike",   "person",
    "pottedplant","sheep",      "sofa",        "train",
    "tvmonitor"
]

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print(f"  Total Classes    : {len(CLASSES) - 1} (excluding background)")
print()
print("  Detectable Objects:")
for i, cls in enumerate(CLASSES[1:], 1):
    print(f"    {i:2}. {cls}")
print()
print("✅ Class labels loaded!")
print()

# ============================================================
# STEP 2: LOAD YOUR OWN IMAGE
# ============================================================

print("[ STEP 2 ] Loading Your Image...")
print("-" * 60)

# Build the full path to your image
test_img_path = os.path.join(IMAGES_DIR, MY_IMAGE_FILENAME)

# Check if the file actually exists
if not os.path.exists(test_img_path):
    print(f"  ❌ ERROR: Image not found!")
    print(f"     Expected path : {test_img_path}")
    print()
    print(f"  HOW TO FIX:")
    print(f"  1. Copy your photo into the images/ folder")
    print(f"  2. Rename it to: {MY_IMAGE_FILENAME}")
    print(f"  3. OR update MY_IMAGE_FILENAME at the")
    print(f"     top of this script to match your filename")
    print()
    print(f"  Example: MY_IMAGE_FILENAME = 'street.jpg'")
    exit()

# Load the image with OpenCV to verify it's readable
verify = cv2.imread(test_img_path)
if verify is None:
    print(f"  ❌ ERROR: File exists but cannot be read!")
    print(f"     Make sure it is a valid JPG or PNG image")
    exit()

# Success — print image info
h_img, w_img = verify.shape[:2]
file_size_kb  = os.path.getsize(test_img_path) / 1024

print(f"  ✅ Image loaded successfully!")
print(f"     File      : {MY_IMAGE_FILENAME}")
print(f"     Full path : {test_img_path}")
print(f"     Dimensions: {w_img} x {h_img} pixels")
print(f"     File size : {file_size_kb:.1f} KB")
print()
print("✅ Image is ready for detection!")
print()

# ============================================================
# STEP 3: LOAD THE MOBILENET-SSD MODEL
# ============================================================

print("[ STEP 3 ] Loading MobileNet-SSD Model...")
print("-" * 60)

PROTOTXT_PATH   = os.path.join(MODELS_DIR, 'MobileNetSSD_deploy.prototxt')
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, 'MobileNetSSD_deploy.caffemodel')

print("  Checking model files...")

if not os.path.exists(PROTOTXT_PATH):
    print(f"  ❌ Missing: {PROTOTXT_PATH}")
    print(f"     Download it and place in models/ folder")
    exit()
else:
    size = os.path.getsize(PROTOTXT_PATH)
    print(f"  ✅ Found : MobileNetSSD_deploy.prototxt ({size:,} bytes)")

if not os.path.exists(CAFFEMODEL_PATH):
    print(f"  ❌ Missing: {CAFFEMODEL_PATH}")
    print(f"     Download it and place in models/ folder")
    exit()
else:
    size = os.path.getsize(CAFFEMODEL_PATH)
    print(f"  ✅ Found : MobileNetSSD_deploy.caffemodel ({size/1024/1024:.1f} MB)")

print()
print("  Loading model into OpenCV DNN...")

try:
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)
    layer_names = net.getLayerNames()
    print(f"  ✅ Model loaded successfully!")
    print()
    print(f"  Model Architecture :")
    print(f"    Framework        : Caffe")
    print(f"    Total Layers     : {len(layer_names)}")
    print(f"    Input Size       : 300 x 300 pixels")
    print(f"    Output Classes   : {len(CLASSES) - 1}")
    print(f"    Input Mean       : 127.5")
    print(f"    Scale Factor     : 0.007843")

except Exception as e:
    print(f"  ❌ Model loading failed: {e}")
    exit()

print()
print("✅ Model ready for inference!")
print()

# ============================================================
# STEP 4: RUN OBJECT DETECTION
# ============================================================

print("[ STEP 4 ] Running Object Detection...")
print("-" * 60)


def detect_objects(image_path, net, confidence_threshold=0.5):
    """
    Run MobileNet-SSD object detection on an image.

    Parameters:
        image_path           : str   → Path to input image
        net                  : DNN   → Loaded OpenCV model
        confidence_threshold : float → Min score to accept (0-1)

    Returns:
        tuple: (annotated_image, detections_list)
    """

    # --- Load image ---
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    print(f"  Image dimensions : {w} x {h} pixels")
    print()

    # ── PRE-PROCESSING ────────────────────────────────────
    # Explanation:
    # The network was trained on 300x300 images
    # with pixel values normalized around 0
    # blobFromImage does all of this automatically:
    #   1. Resize to 300x300
    #   2. Subtract mean 127.5 from each channel
    #   3. Multiply by scale factor 0.007843
    # Result: values roughly in range -1.0 to +1.0

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),  # Step 1: resize
        0.007843,                        # Step 2: scale
        (300, 300),                      # Target size
        127.5                            # Step 3: mean
    )

    print("  Pre-processing   :")
    print(f"    Original size  : {w} x {h}")
    print(f"    Resized to     : 300 x 300")
    print(f"    Mean subtract  : 127.5")
    print(f"    Scale factor   : 0.007843")
    print(f"    Blob shape     : {blob.shape}")
    print()

    # ── FORWARD PASS ─────────────────────────────────────
    # Feed the blob into the network
    # net.forward() runs the entire neural network
    # and returns all candidate detections
    print("  Running forward pass through network...")
    net.setInput(blob)
    detections = net.forward()
    print(f"  Raw candidate boxes : {detections.shape[2]}")
    print()

    # ── FILTER AND DRAW ───────────────────────────────────
    results      = []
    output_image = image.copy()

    print(f"  Filtering by confidence >= "
          f"{confidence_threshold * 100:.0f}%...")
    print()

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Skip anything below our threshold
        if confidence < confidence_threshold:
            continue

        class_idx   = int(detections[0, 0, i, 1])
        class_label = CLASSES[class_idx]

        if class_label == "background":
            continue

        # Network gives normalized coords (0.0 to 1.0)
        # Multiply by actual image size to get pixel coords
        box      = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        # Clamp to image boundaries
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w, x2);  y2 = min(h, y2)

        color      = COLORS[class_idx].tolist()
        label_text = f"{class_label}: {confidence * 100:.1f}%"

        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(
            output_image,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color, -1
        )

        # Draw label text (white)
        cv2.putText(
            output_image, label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2
        )

        results.append({
            'class'     : class_label,
            'confidence': float(confidence),
            'box'       : (x1, y1, x2, y2)
        })

        print(f"  ✅ DETECTED : {class_label:<15}"
              f" | Confidence: {confidence * 100:.1f}%"
              f" | Box: ({x1},{y1})-({x2},{y2})")

    print()
    return output_image, results


def save_detection_output(original_path, output_image, results):
    """
    Save the side-by-side comparison and print summary.
    """

    original = cv2.imread(original_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image\n(Your Input Photo)',
                      fontweight='bold', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f'Detection Results — {len(results)} Object(s) Found',
        fontweight='bold', fontsize=12)
    axes[1].axis('off')

    plt.suptitle(
        'MobileNet-SSD Object Detection\n'
        'DecodeLabs Project 4 | Validation Rule 4 ✅',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    # Save the side-by-side result
    output_path = os.path.join(OUTPUTS_DIR, 'detection_result.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Also save just the annotated image separately
    cv2.imwrite(
        os.path.join(OUTPUTS_DIR, 'detection_annotated.jpg'),
        output_image
    )
    print(f"  ✅ Saved : outputs/detection_result.png")
    print(f"  ✅ Saved : outputs/detection_annotated.jpg")
    print()

    # --- Summary and Validation Report ---
    if results:
        avg_conf = np.mean([r['confidence'] for r in results]) * 100
        max_conf = max([r['confidence'] for r in results]) * 100

        print("  DETECTION SUMMARY:")
        print(f"  {'─' * 45}")
        print(f"  Objects found    : {len(results)}")
        print(f"  Average conf     : {avg_conf:.1f}%")
        print(f"  Best detection   : {max_conf:.1f}%")
        print()

        print(f"  {'Object':<20} {'Confidence':>12}  {'Bar':}")
        print(f"  {'─' * 50}")
        for r in sorted(results,
                        key=lambda x: x['confidence'],
                        reverse=True):
            bar = "█" * int(r['confidence'] * 20)
            print(f"  {r['class']:<20}"
                  f" {r['confidence'] * 100:>8.1f}%"
                  f"  {bar}")
        print()

        rule3_status = "✅ PASSED" if avg_conf >= 80 else "⚠️  BELOW 80% — try a clearer photo"

        print("=" * 60)
        print("  FINAL VALIDATION REPORT")
        print("=" * 60)
        print(f"  Rule 1 — Library Integration  : ✅ PASSED")
        print(f"           cv2.dnn model loaded successfully")
        print(f"  Rule 2 — Pre-Processing        : ✅ PASSED")
        print(f"           Blob 300x300, mean=127.5, scale=0.007843")
        print(f"  Rule 3 — Confidence >= 80%     : {rule3_status}")
        print(f"           Average score: {avg_conf:.1f}%")
        print(f"  Rule 4 — Visual Confirmation   : ✅ PASSED")
        print(f"           detection_result.png saved")
        print("=" * 60)
        print()
        print("🎉 PATH 2 (OBJECT DETECTION) COMPLETE!")
        print("=" * 60)

    else:
        print("  ⚠️  No objects detected above 50% threshold.")
        print()
        print("  Possible reasons:")
        print("  1. The image may not contain detectable objects")
        print("     (aeroplane, bicycle, bird, boat, bottle, bus,")
        print("      car, cat, chair, cow, dog, horse, motorbike,")
        print("      person, sheep, sofa, train, tvmonitor)")
        print("  2. The image may be too dark, blurry, or small")
        print("  3. Objects may be too far away or partially cut off")
        print()
        print("  SOLUTION: Replace your image with a clearer photo")
        print(f"  containing the objects listed above.")
        print(f"  Place it in images/ and update MY_IMAGE_FILENAME.")


# --- Run the full pipeline ---
output_img, detections = detect_objects(
    test_img_path, net,
    confidence_threshold=0.5
)

if output_img is not None:
    save_detection_output(test_img_path, output_img, detections)