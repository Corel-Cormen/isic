import argparse
import numpy as np
import os
import re
import shutil
import subprocess
import tempfile
import tensorflow as tf
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
GENERATED_FILE_SUFFIXES = ("_hair.png", "_mask.png", "_alpha.png")


@dataclass(frozen=True)
class HairUnetConfig:
    image_size: tuple[int, int] = (384, 384)
    batch_size: int = 4
    epochs: int = 50
    frozen_epochs: int = 15
    validation_split: float = 0.2
    learning_rate: float = 3e-4
    fine_tune_learning_rate: float = 3e-5
    fine_tune_at: int = 110
    encoder_weights: str | None = "imagenet"
    seed: int = 42
    model_path: str = "models/HairMaskDetermiNet.keras"
    metrics_csv_path: str = "models/HairMaskDetermiNet_metrics.csv"
    prediction_threshold: float = 0.5
    regenerate_each_epoch: bool = True


def generate_data(hair_path: str, hair_mask_path: str, seed: int, source_ids: set[str] | None = None) -> None:
    """Wygeneruj dane i atomowo podmień pliki wybranych obrazów bazowych."""
    if not os.path.exists(hair_path):
        subprocess.run(
            [str(Path(__file__).resolve().parent / "hair_select.sh")],
            cwd=PROJECT_DIR,
            check=True,
        )
    output = Path(hair_mask_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hair_mask_", dir=output.parent) as temporary_dir:
        temporary_root = Path(temporary_dir)
        temporary_output = temporary_root / "output"
        temporary_output.mkdir()

        generator_input = Path(hair_path)
        if source_ids is not None:
            temporary_input = temporary_root / "input"
            temporary_input.mkdir()
            selected_images = [
                path
                for path in generator_input.glob("*.jpg")
                if _source_image_id(str(path)) in source_ids
            ]
            if not selected_images:
                raise ValueError("Nie znaleziono obrazów treningowych dla wybranych ID.")
            for image_path in selected_images:
                target = temporary_input / image_path.name
                try:
                    os.link(image_path, target)
                except OSError:
                    shutil.copy2(image_path, target)
            generator_input = temporary_input

        subprocess.run(
            [
                str(PROJECT_DIR / "tools" / "hair_bezier_generator" / "build" / "bin" / "hair_bezier_generator"),
                "--input-dir", str(generator_input),
                "--output-dir", str(temporary_output),
                "--seed", str(seed),
            ],
            check=True
        )

        generated_files = [
            path
            for path in temporary_output.iterdir()
            if path.is_file() and path.name.endswith(GENERATED_FILE_SUFFIXES)
        ]
        if source_ids is not None:
            generated_files = [
                path for path in generated_files if _source_image_id(str(path)) in source_ids
            ]
        if not generated_files:
            raise ValueError("Generator nie utworzył plików dla wybranych obrazów bazowych.")

        generated_names = {path.name for path in generated_files}
        for old_path in output.iterdir():
            if not old_path.is_file() or not old_path.name.endswith(GENERATED_FILE_SUFFIXES):
                continue
            if source_ids is not None and _source_image_id(str(old_path)) not in source_ids:
                continue
            if old_path.name not in generated_names:
                old_path.unlink()

        for generated_path in generated_files:
            os.replace(generated_path, output / generated_path.name)


def collect_image_pairs(hair_mask_path: str) -> list[tuple[str, str]]:
    output = Path(hair_mask_path)
    hair_files = {path.stem.removesuffix("_hair"): path for path in output.glob("*_hair.png")}
    mask_files = {path.stem.removesuffix("_mask"): path for path in output.glob("*_mask.png")}
    alpha_files = {path.stem.removesuffix("_alpha"): path for path in output.glob("*_alpha.png")}

    missing_masks = sorted(set(hair_files) - set(mask_files))
    missing_images = sorted(set(mask_files) - set(hair_files))
    missing_alphas = sorted(set(hair_files) - set(alpha_files))
    if missing_masks or missing_images or missing_alphas:
        raise ValueError(
            "Niepełne pary obrazów: "
            f"brak masek={len(missing_masks)}, brak obrazów={len(missing_images)}, "
            f"brak masek alfa={len(missing_alphas)}."
        )

    pairs = [(str(hair_files[name]), str(mask_files[name])) for name in sorted(hair_files)]
    if not pairs:
        raise ValueError(f"Nie znaleziono par *_hair.png i *_mask.png w {hair_mask_path}.")
    return pairs


def _source_image_id(image_path: str) -> str:
    """Zwróć ID obrazu bazowego, wspólne dla jego syntetycznych wariantów."""
    match = re.match(r"^(ISIC_\d+)_", Path(image_path).name)
    return match.group(1) if match else Path(image_path).stem.removesuffix("_hair")


def split_pairs_by_source(pairs: list[tuple[str, str]], validation_split: float, seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split musi należeć do przedziału (0, 1).")

    groups: dict[str, list[tuple[str, str]]] = {}
    for pair in pairs:
        groups.setdefault(_source_image_id(pair[0]), []).append(pair)
    if len(groups) < 2:
        raise ValueError("Do podziału train/validation potrzebne są co najmniej 2 obrazy bazowe.")

    group_ids = np.array(sorted(groups))
    np.random.default_rng(seed).shuffle(group_ids)
    validation_count = min(len(group_ids) - 1, max(1, int(round(len(group_ids) * validation_split))),)
    validation_ids = set(group_ids[:validation_count].tolist())
    train_pairs = [pair for group_id in group_ids if group_id not in validation_ids for pair in groups[group_id]]
    validation_pairs = [pair for group_id in group_ids if group_id in validation_ids for pair in groups[group_id]]
    return train_pairs, validation_pairs


def _load_pair(
    image_path: tf.Tensor,
    mask_path: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(tf.io.read_file(image_path), channels=3, expand_animations=False)
    mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
    alpha_path = tf.strings.regex_replace(mask_path, "_mask\\.png$", "_alpha.png")
    alpha = tf.io.decode_image(tf.io.read_file(alpha_path), channels=1, expand_animations=False)
    image.set_shape([None, None, 3])
    mask.set_shape([None, None, 1])
    alpha.set_shape([None, None, 1])

    image = tf.image.resize(image, image_size, method="bilinear", antialias=True)
    image = tf.cast(image, tf.float32) / 255.0

    # AREA zachowuje informację o bardzo cienkiej linii podczas zmniejszania.
    # Próg 0.05 oznacza: zachowaj piksel, jeśli włos pokrywa co najmniej 5% jego
    # powierzchni. NEAREST potrafił całkowicie zgubić włos między próbkami.
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.image.resize(mask, image_size, method="area", antialias=True)
    mask = tf.cast(mask >= 0.05, tf.float32)
    # Miękka alfa pozwala zmieniać kolor włosa bez tworzenia sztucznej,
    # ostrej i grubszej linii na brzegu.
    alpha = tf.cast(alpha, tf.float32) / 255.0
    alpha = tf.image.resize(alpha, image_size, method="bilinear", antialias=True)
    return image, mask, tf.clip_by_value(alpha, 0.0, 1.0)


def _augment(
    image: tf.Tensor, mask: tf.Tensor, alpha: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        alpha = tf.image.flip_left_right(alpha)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
        alpha = tf.image.flip_up_down(alpha)

    rotations = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, rotations)
    mask = tf.image.rot90(mask, rotations)
    alpha = tf.image.rot90(alpha, rotations)

    # Około połowa próbek zawiera neutralny siwy lub biały włos. Nakładamy
    # kolor przez miękką alfę generatora, więc również linie subpikselowe i ich
    # antyaliasowane brzegi pozostają realistyczne.
    if tf.random.uniform(()) < 0.55:
        brightness = tf.random.uniform((), 0.72, 0.98)
        tint = tf.random.uniform((3,), -0.035, 0.035)
        pale_hair = tf.clip_by_value(brightness + tint, 0.65, 1.0)
        recolor_strength = tf.random.uniform((), 0.75, 1.25)
        blend = tf.clip_by_value(alpha * recolor_strength, 0.0, 1.0)
        image = image * (1.0 - blend) + pale_hair * blend

    # Symuluje także ciemne zdjęcia i włosy o małym kontraście względem tła.
    image = tf.image.adjust_brightness(image, tf.random.uniform((), -0.16, 0.08))
    image = tf.image.adjust_contrast(image, tf.random.uniform((), 0.65, 1.35))
    # adjust_gamma(x, gamma) oblicza x**gamma. Po brightness/contrast część
    # wartości może być ujemna, co dla niecałkowitego gamma dawało NaN.
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.adjust_gamma(image, tf.random.uniform((), 0.70, 1.45))
    image = tf.image.adjust_saturation(image, tf.random.uniform((), 0.80, 1.20))
    return tf.clip_by_value(image, 0.0, 1.0), mask


def build_datasets(pairs: list[tuple[str, str]], config: HairUnetConfig) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_pairs, validation_pairs = split_pairs_by_source(pairs, config.validation_split, config.seed)

    def make_dataset(items: list[tuple[str, str]], training: bool) -> tf.data.Dataset:
        image_paths, mask_paths = zip(*items)
        dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))
        if training:
            dataset = dataset.shuffle(len(items), seed=config.seed, reshuffle_each_iteration=True)
        dataset = dataset.map(
            lambda image, mask: _load_pair(image, mask, config.image_size),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if training:
            dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.map(
                lambda image, mask, alpha: (image, mask),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return make_dataset(train_pairs, True), make_dataset(validation_pairs, False)


def _conv_block(inputs: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def _decoder_block(inputs: tf.Tensor, skip: tf.Tensor, filters: int) -> tf.Tensor:
    x = tf.keras.layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip])
    return _conv_block(x, filters)


def weighted_bce_tversky_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Strata dla rzadkich, cienkich obiektów, z większą karą za false negative."""
    # Obliczenia w float32 są konieczne przy mixed_float16. W float16 wartość
    # 1-epsilon zaokrąglała się do 1, więc sigmoid=1 prowadził do log(0).
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.constant(1e-7, dtype=tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    positive_weight = 4.0
    weighted_bce = -(
        positive_weight * y_true * tf.math.log(y_pred)
        + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )

    axes = (1, 2, 3)
    true_positive = tf.reduce_sum(y_true * y_pred, axis=axes)
    false_positive = tf.reduce_sum((1.0 - y_true) * y_pred, axis=axes)
    false_negative = tf.reduce_sum(y_true * (1.0 - y_pred), axis=axes)
    # beta > alpha: pominięty włos kosztuje więcej niż nadmiarowy piksel maski.
    alpha, beta = 0.30, 0.70
    tversky = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    focal_tversky = tf.pow(1.0 - tversky, 0.75)
    return tf.reduce_mean(weighted_bce) + tf.reduce_mean(focal_tversky)


def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.cast(y_pred >= HairUnetConfig.prediction_threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return tf.reduce_mean((2.0 * intersection + 1.0) / (denominator + 1.0))


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=weighted_bce_tversky_loss,
        metrics=[
            dice_coefficient,
            tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5),
            tf.keras.metrics.Recall(thresholds=0.5, name="hair_recall"),
        ],
    )


def build_unet(config: HairUnetConfig) -> tf.keras.Model:
    """U-Net z encoderem MobileNetV2 wytrenowanym na ImageNet."""
    height, width = config.image_size
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("image_size musi być podzielne przez 32, np. (320, 320).")

    inputs = tf.keras.layers.Input((*config.image_size, 3), name="hair_image")
    encoder_input = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="imagenet_preprocessing")(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*config.image_size, 3),
        include_top=False,
        weights=config.encoder_weights,
    )
    feature_layer_names = (
        "block_1_expand_relu",   # 1/2
        "block_3_expand_relu",   # 1/4
        "block_6_expand_relu",   # 1/8
        "block_13_expand_relu",  # 1/16
        "block_16_project",      # 1/32
    )
    encoder = tf.keras.Model(
        backbone.input,
        [backbone.get_layer(name).output for name in feature_layer_names],
        name="mobilenetv2_encoder",
    )
    encoder.trainable = False

    skip_2, skip_4, skip_8, skip_16, x = encoder(encoder_input, training=False)
    x = _decoder_block(x, skip_16, 256)
    x = _decoder_block(x, skip_8, 128)
    x = _decoder_block(x, skip_4, 64)
    x = _decoder_block(x, skip_2, 32)

    # Pełnorozdzielczy skip przekazuje krawędzie, których encoder 1/2 mógł nie zachować.
    detail_skip = _conv_block(inputs, 16)
    x = tf.keras.layers.Conv2DTranspose(16, 2, strides=2, padding="same")(x)
    x = tf.keras.layers.Concatenate()([x, detail_skip])
    x = _conv_block(x, 16)

    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="hair_mask")(x)
    model = tf.keras.Model(inputs, outputs, name="hair_mobilenetv2_unet")
    compile_model(model, config.learning_rate)
    return model


def _training_callbacks(
    model_path: str,
    metrics_csv_path: str,
    initial_best: float | None = None,
    append_csv: bool = False,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(metrics_csv_path, append=append_csv),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            initial_value_threshold=initial_best,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient", mode="max", factor=0.5, patience=3, min_lr=1e-7
        ),
    ]


class RegenerateTrainingData(tf.keras.callbacks.Callback):
    """Tworzy nowe syntetyczne warianty treningowe przed każdą epoką."""

    def __init__(
        self,
        hair_path: str,
        hair_mask_path: str,
        source_ids: set[str],
        base_seed: int,
    ) -> None:
        super().__init__()
        self.hair_path = hair_path
        self.hair_mask_path = hair_mask_path
        self.source_ids = source_ids
        self.base_seed = base_seed

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        epoch_seed = self.base_seed + epoch + 1
        print(f"\nGenerowanie nowych danych treningowych dla epoki {epoch + 1} (seed={epoch_seed})...")
        generate_data(
            self.hair_path,
            self.hair_mask_path,
            epoch_seed,
            source_ids=self.source_ids,
        )


def train_hair_unet(
    hair_path: str = str(PROJECT_DIR / "dataset" / "hair"),
    hair_mask_path: str = str(PROJECT_DIR / "dataset" / "hair_mask"),
    config: HairUnetConfig | None = None,
) -> tf.keras.callbacks.History:
    config = config or HairUnetConfig()
    tf.keras.utils.set_random_seed(config.seed)
    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.metrics_csv_path).parent.mkdir(parents=True, exist_ok=True)
    generate_data(hair_path, hair_mask_path, config.seed)
    pairs = collect_image_pairs(hair_mask_path)
    train_pairs, _ = split_pairs_by_source(pairs, config.validation_split, config.seed)
    train_source_ids = {_source_image_id(image_path) for image_path, _ in train_pairs}
    train_dataset, validation_dataset = build_datasets(pairs, config)
    model = build_unet(config)

    def callbacks(
        initial_best: float | None = None,
        append_csv: bool = False,
    ) -> list[tf.keras.callbacks.Callback]:
        result = _training_callbacks(
            config.model_path,
            config.metrics_csv_path,
            initial_best=initial_best,
            append_csv=append_csv,
        )
        if config.regenerate_each_epoch:
            result.insert(
                0,
                RegenerateTrainingData(
                    hair_path,
                    hair_mask_path,
                    train_source_ids,
                    config.seed,
                ),
            )
        return result

    warmup_epochs = min(config.frozen_epochs, config.epochs)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=warmup_epochs,
        callbacks=callbacks(),
    )

    if warmup_epochs < config.epochs:
        # Otwórz tylko końcową część encodera. BatchNormalization pozostaje
        # zamrożony; przy małym batchu jego statystyki byłyby niestabilne.
        encoder = model.get_layer("mobilenetv2_encoder")
        encoder.trainable = True
        for index, layer in enumerate(encoder.layers):
            layer.trainable = index >= config.fine_tune_at and not isinstance(
                layer, tf.keras.layers.BatchNormalization
            )
        compile_model(model, config.fine_tune_learning_rate)

        baseline = float(
            model.evaluate(validation_dataset, verbose=0, return_dict=True)["dice_coefficient"]
        )
        fine_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            initial_epoch=warmup_epochs,
            epochs=config.epochs,
            callbacks=callbacks(initial_best=baseline, append_csv=True),
        )
        for key, values in history.history.items():
            fine_history.history[key] = values + fine_history.history.get(key, [])
        fine_history.epoch = history.epoch + fine_history.epoch
        history = fine_history

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net training for determinet hair mask.")
    parser.add_argument("--epochs", type=int, default=HairUnetConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=HairUnetConfig.batch_size)
    parser.add_argument("--image-size", type=int, default=HairUnetConfig.image_size[0])
    parser.add_argument("--frozen-epochs", type=int, default=HairUnetConfig.frozen_epochs)
    parser.add_argument("--fine-tune-at", type=int, default=HairUnetConfig.fine_tune_at)
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--model-path", default=str(PROJECT_DIR / HairUnetConfig.model_path))
    parser.add_argument("--metrics-csv", default=str(PROJECT_DIR / HairUnetConfig.metrics_csv_path))
    parser.add_argument("--generate-once", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_hair_unet(
        config=HairUnetConfig(
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            frozen_epochs=args.frozen_epochs,
            fine_tune_at=args.fine_tune_at,
            encoder_weights=None if args.no_imagenet else "imagenet",
            model_path=args.model_path,
            metrics_csv_path=args.metrics_csv,
            regenerate_each_epoch=not args.generate_once,
        )
    )
