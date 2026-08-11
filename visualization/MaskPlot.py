import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path


def show_mask_prediction(model, image_path, threshold=0.5, alpha=0.55):
    """Nałóż ocenę predykcji modelu na obraz z img_hair_list.

    Zielony: poprawnie wykryty włos (TP).
    Czerwony: fałszywie wykryty lub pominięty włos (FP albo FN).
    Funkcja automatycznie odnajduje odpowiadającą maskę *_mask.png.
    """
    image_path = Path(image_path)
    if not image_path.name.lower().endswith("_hair.png"):
        raise ValueError("Oczekiwano obrazu o nazwie kończącej się na '_hair.png'.")

    mask_path = image_path.with_name(image_path.name[:-9] + "_mask.png")
    if not mask_path.exists():
        raise FileNotFoundError(f"Nie znaleziono maski wzorcowej: {mask_path}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    true_mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or true_mask_raw is None:
        raise ValueError("Nie udało się odczytać obrazu lub maski wzorcowej.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred_mask = model.predict(str(image_path), threshold=threshold) > 0
    true_mask = true_mask_raw > 127

    if true_mask.shape != pred_mask.shape:
        true_mask = cv2.resize(
            true_mask.astype(np.uint8),
            (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    true_positive = pred_mask & true_mask
    errors = pred_mask ^ true_mask  # FP i FN

    colors = np.zeros_like(image_rgb)
    colors[true_positive] = (0, 255, 0)
    colors[errors] = (255, 0, 0)

    overlay = image_rgb.copy()
    colored_pixels = true_positive | errors
    overlay[colored_pixels] = (
        (1 - alpha) * image_rgb[colored_pixels] + alpha * colors[colored_pixels]
    ).astype(np.uint8)

    intersection = np.count_nonzero(pred_mask & true_mask)
    union = np.count_nonzero(pred_mask | true_mask)
    mask_sum = np.count_nonzero(pred_mask) + np.count_nonzero(true_mask)
    dice = 1.0 if mask_sum == 0 else 2.0 * intersection / mask_sum
    iou = 1.0 if union == 0 else intersection / union

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(overlay)
    ax.set_title(f"{image_path.name} | Dice: {dice:.3f} | IoU: {iou:.3f}")
    ax.axis("off")
    ax.legend(
        handles=[
            Patch(facecolor="lime", label="Poprawnie wykryte włosy (TP)"),
            Patch(facecolor="red", label="Błędna predykcja (FP lub FN)"),
        ],
        loc="lower right",
    )
    plt.tight_layout()
    plt.show()

    return {
        "dice": dice,
        "iou": iou,
        "predicted_mask": pred_mask,
        "true_mask": true_mask,
        "overlay": overlay,
    }


def show_real_image_prediction(model, image_path, threshold=0.5, alpha=0.55):
    """Pokaż oryginał, predykcję maski i maskę nałożoną na prawdziwe zdjęcie."""
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Nie udało się odczytać obrazu: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predicted_mask = model.predict(str(image_path), threshold=threshold) > 0

    if predicted_mask.shape != image_rgb.shape[:2]:
        predicted_mask = cv2.resize(
            predicted_mask.astype(np.uint8),
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    overlay = image_rgb.copy()
    green = np.array([0, 255, 0], dtype=np.float32)
    overlay[predicted_mask] = (
        (1 - alpha) * image_rgb[predicted_mask] + alpha * green
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("1. Oryginalny obraz")

    axes[1].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"2. Predykcja maski (próg: {threshold})")

    axes[2].imshow(overlay)
    axes[2].set_title("3. Maska nałożona na obraz")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(image_path.name, fontsize=14)
    plt.tight_layout()
    plt.show()

    return {
        "predicted_mask": predicted_mask,
        "overlay": overlay,
    }
