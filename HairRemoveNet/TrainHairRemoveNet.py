"""Trening U-Neta odtwarzajacego skore zakryta syntetycznymi wlosami.

Wejscie modelu sklada sie z obrazu RGB ``*_hair.png`` i binarnej maski
``*_mask.png``. Obrazem docelowym jest odpowiadajacy im, czysty obraz z
``dataset/hair``.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from HairRemoveNet.TrainMaskDetermiNet import RegenerateTrainingData, generate_data


PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TrainingTriplet = tuple[str, str, str]


@dataclass(frozen=True)
class HairRemovalUnetConfig:
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
    model_path: str = "models/HairRemoveNet.keras"
    metrics_csv_path: str = "models/HairRemoveNet_metrics.csv"
    regenerate_each_epoch: bool = True
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4


def source_image_id(path: str | Path) -> str:
    """Zwroc ID czystego obrazu wspolne dla wszystkich jego wariantow."""
    name = Path(path).stem
    match = re.match(r"^(ISIC_\d+)(?:_|$)", name)
    if match:
        return match.group(1)
    return name.removesuffix("_hair").removesuffix("_mask")


def collect_training_triplets(hair_mask_path: str, clean_image_path: str) -> list[TrainingTriplet]:
    """Polacz ``*_hair``, ``*_mask`` i bazowy czysty obraz w trojki."""
    generated_dir = Path(hair_mask_path)
    clean_dir = Path(clean_image_path)

    hair_files = {
        path.stem.removesuffix("_hair"): path
        for path in generated_dir.glob("*_hair.png")
    }
    mask_files = {
        path.stem.removesuffix("_mask"): path
        for path in generated_dir.glob("*_mask.png")
    }

    missing_masks = sorted(set(hair_files) - set(mask_files))
    missing_hair_images = sorted(set(mask_files) - set(hair_files))
    if missing_masks or missing_hair_images:
        raise ValueError(
            "Niepelne pary danych syntetycznych: "
            f"brak masek={len(missing_masks)}, brak obrazow z wlosami={len(missing_hair_images)}."
        )

    clean_files: dict[str, Path] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in clean_dir.glob(f"*{extension}"):
            image_id = source_image_id(path)
            if image_id in clean_files:
                raise ValueError(f"Wiele czystych obrazow ma ID {image_id}: {clean_files[image_id]} i {path}.")
            clean_files[image_id] = path

    triplets: list[TrainingTriplet] = []
    missing_targets: set[str] = set()
    for variant_name in sorted(hair_files):
        image_id = source_image_id(hair_files[variant_name])
        target = clean_files.get(image_id)
        if target is None:
            missing_targets.add(image_id)
            continue
        triplets.append(
            (str(hair_files[variant_name]), str(mask_files[variant_name]), str(target))
        )

    if missing_targets:
        preview = ", ".join(sorted(missing_targets)[:5])
        raise ValueError(
            f"Brak czystych obrazow docelowych dla {len(missing_targets)} ID: {preview}."
        )
    if not triplets:
        raise ValueError(
            f"Nie znaleziono trojek *_hair.png, *_mask.png i czystego obrazu w "
            f"{generated_dir} oraz {clean_dir}."
        )
    return triplets


def split_triplets_by_source(
    triplets: list[TrainingTriplet], validation_split: float, seed: int
) -> tuple[list[TrainingTriplet], list[TrainingTriplet]]:
    """Dziel po obrazie bazowym, a nie po syntetycznym wariancie."""
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split musi nalezec do przedzialu (0, 1).")

    groups: dict[str, list[TrainingTriplet]] = {}
    for triplet in triplets:
        groups.setdefault(source_image_id(triplet[0]), []).append(triplet)
    if len(groups) < 2:
        raise ValueError("Do podzialu train/validation potrzebne sa co najmniej 2 obrazy bazowe.")

    group_ids = np.array(sorted(groups))
    np.random.default_rng(seed).shuffle(group_ids)
    validation_count = min(
        len(group_ids) - 1,
        max(1, int(round(len(group_ids) * validation_split))),
    )
    validation_ids = set(group_ids[:validation_count].tolist())
    train = [item for image_id in group_ids if image_id not in validation_ids for item in groups[image_id]]
    validation = [item for image_id in group_ids if image_id in validation_ids for item in groups[image_id]]
    return train, validation


def _load_triplet(
    hair_image_path: tf.Tensor,
    mask_path: tf.Tensor,
    target_path: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    hair_image = tf.io.decode_image(
        tf.io.read_file(hair_image_path), channels=3, expand_animations=False
    )
    mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1, expand_animations=False)
    target = tf.io.decode_image(tf.io.read_file(target_path), channels=3, expand_animations=False)
    hair_image.set_shape([None, None, 3])
    mask.set_shape([None, None, 1])
    target.set_shape([None, None, 3])

    hair_image = tf.image.resize(hair_image, image_size, method="bilinear", antialias=True)
    target = tf.image.resize(target, image_size, method="bilinear", antialias=True)
    hair_image = tf.cast(hair_image, tf.float32) / 255.0
    target = tf.cast(target, tf.float32) / 255.0

    # AREA nie gubi cienkich wlosow przy zmniejszaniu obrazu. Niski prog
    # zachowuje takze piksele krawedziowe, na ktore wplynal antyaliasing.
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.image.resize(mask, image_size, method="area", antialias=True)
    mask = tf.cast(mask >= 0.05, tf.float32)
    return hair_image, mask, target


def _augment_triplet(
    hair_image: tf.Tensor, mask: tf.Tensor, target: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    if tf.random.uniform(()) > 0.5:
        hair_image = tf.image.flip_left_right(hair_image)
        mask = tf.image.flip_left_right(mask)
        target = tf.image.flip_left_right(target)
    if tf.random.uniform(()) > 0.5:
        hair_image = tf.image.flip_up_down(hair_image)
        mask = tf.image.flip_up_down(mask)
        target = tf.image.flip_up_down(target)

    # Obrot o 180 stopni dziala takze dla niekwadratowego image_size.
    if tf.random.uniform(()) > 0.5:
        hair_image = tf.image.rot90(hair_image, 2)
        mask = tf.image.rot90(mask, 2)
        target = tf.image.rot90(target, 2)
    return hair_image, mask, target


def _pack_example(
    hair_image: tf.Tensor, mask: tf.Tensor, target: tf.Tensor
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    # Czwarty kanal y_true przechowuje maske dla funkcji straty i metryk.
    target_with_mask = tf.concat([target, mask], axis=-1)
    return {"hair_image": hair_image, "hair_mask": mask}, target_with_mask


def build_datasets(
    triplets: list[TrainingTriplet], config: HairRemovalUnetConfig
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    train_triplets, validation_triplets = split_triplets_by_source(
        triplets, config.validation_split, config.seed
    )

    def make_dataset(items: list[TrainingTriplet], training: bool) -> tf.data.Dataset:
        hair_paths, mask_paths, target_paths = zip(*items)
        dataset = tf.data.Dataset.from_tensor_slices(
            (list(hair_paths), list(mask_paths), list(target_paths))
        )
        if training:
            dataset = dataset.shuffle(
                len(items), seed=config.seed, reshuffle_each_iteration=True
            )
        dataset = dataset.map(
            lambda hair, mask, target: _load_triplet(
                hair, mask, target, config.image_size
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if training:
            dataset = dataset.map(_augment_triplet, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(_pack_example, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    return make_dataset(train_triplets, True), make_dataset(validation_triplets, False)


def _conv_block(inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_conv1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.Activation("relu", name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    return tf.keras.layers.Activation("relu", name=f"{name}_relu2")(x)


def _mask_at(mask: tf.Tensor, reference: tf.Tensor, name: str) -> tf.Tensor:
    height, width = reference.shape[1:3]
    if height is None or width is None:
        raise ValueError("Rozmiary map cech musza byc znane podczas budowania modelu.")
    return tf.keras.layers.Resizing(
        height, width, interpolation="nearest", name=name
    )(mask)


def _decoder_block(
    inputs: tf.Tensor,
    skip: tf.Tensor,
    mask: tf.Tensor,
    filters: int,
    name: str,
) -> tf.Tensor:
    x = tf.keras.layers.Conv2DTranspose(
        filters, 2, strides=2, padding="same", name=f"{name}_up"
    )(inputs)
    scaled_mask = _mask_at(mask, skip, f"{name}_mask")
    x = tf.keras.layers.Concatenate(name=f"{name}_concat")([x, skip, scaled_mask])
    return _conv_block(x, filters, name)


@tf.keras.utils.register_keras_serializable(package="HairRemoveNet")
def masked_reconstruction_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Charbonnier + SSIM, z naciskiem na obszar wskazany maska."""
    target = tf.cast(y_true[..., :3], tf.float32)
    mask = tf.cast(y_true[..., 3:4], tf.float32)
    prediction = tf.cast(y_pred, tf.float32)

    error = tf.sqrt(tf.square(target - prediction) + 1e-6)
    weights = 1.0 + 20.0 * mask
    weighted_error = tf.reduce_sum(error * weights) / (
        tf.reduce_sum(weights) * tf.cast(tf.shape(target)[-1], tf.float32) + 1e-7
    )
    ssim_loss = 1.0 - tf.reduce_mean(
        tf.image.ssim(target, prediction, max_val=1.0)
    )
    return weighted_error + 0.15 * ssim_loss


@tf.keras.utils.register_keras_serializable(package="HairRemoveNet")
def masked_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    target = tf.cast(y_true[..., :3], tf.float32)
    mask = tf.cast(y_true[..., 3:4], tf.float32)
    prediction = tf.cast(y_pred, tf.float32)
    pixel_error = tf.reduce_mean(tf.abs(target - prediction), axis=-1, keepdims=True)
    return tf.reduce_sum(pixel_error * mask) / (tf.reduce_sum(mask) + 1e-7)


@tf.keras.utils.register_keras_serializable(package="HairRemoveNet")
def masked_psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    target = tf.cast(y_true[..., :3], tf.float32)
    mask = tf.cast(y_true[..., 3:4], tf.float32)
    prediction = tf.cast(y_pred, tf.float32)
    squared_error = tf.reduce_mean(tf.square(target - prediction), axis=-1, keepdims=True)
    mse = tf.reduce_sum(squared_error * mask) / (tf.reduce_sum(mask) + 1e-7)
    return -10.0 * tf.math.log(tf.maximum(mse, 1e-8)) / tf.math.log(10.0)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_reconstruction_loss,
        metrics=[masked_mae, masked_psnr],
    )


def build_unet(config: HairRemovalUnetConfig) -> tf.keras.Model:
    """U-Net z MobileNetV2, maska podana jawnie na kazdym poziomie dekodera."""
    height, width = config.image_size
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("image_size musi byc podzielne przez 32, np. (384, 384).")

    image_input = tf.keras.layers.Input((*config.image_size, 3), name="hair_image")
    mask_input = tf.keras.layers.Input((*config.image_size, 1), name="hair_mask")

    inverse_mask = tf.keras.layers.Rescaling(
        scale=-1.0, offset=1.0, name="inverse_mask"
    )(mask_input)
    visible_image = tf.keras.layers.Multiply(name="visible_image")(
        [image_input, inverse_mask]
    )
    neutral_hole = tf.keras.layers.Rescaling(
        scale=0.5, name="neutral_masked_area"
    )(mask_input)
    encoder_image = tf.keras.layers.Add(name="encoder_image")(
        [visible_image, neutral_hole]
    )
    encoder_input = tf.keras.layers.Rescaling(
        2.0, offset=-1.0, name="imagenet_preprocessing"
    )(encoder_image)

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
    bottleneck_mask = _mask_at(mask_input, x, "bottleneck_mask")
    x = tf.keras.layers.Concatenate(name="bottleneck_concat")([x, bottleneck_mask])
    x = _conv_block(x, 320, "bottleneck")
    x = _decoder_block(x, skip_16, mask_input, 256, "decoder_16")
    x = _decoder_block(x, skip_8, mask_input, 128, "decoder_8")
    x = _decoder_block(x, skip_4, mask_input, 64, "decoder_4")
    x = _decoder_block(x, skip_2, mask_input, 32, "decoder_2")

    detail_input = tf.keras.layers.Concatenate(name="detail_input")(
        [image_input, mask_input]
    )
    detail_skip = _conv_block(detail_input, 16, "detail")
    x = tf.keras.layers.Conv2DTranspose(
        16, 2, strides=2, padding="same", name="decoder_full_up"
    )(x)
    x = tf.keras.layers.Concatenate(name="decoder_full_concat")(
        [x, detail_skip, mask_input]
    )
    x = _conv_block(x, 16, "decoder_full")

    restored_candidate = tf.keras.layers.Conv2D(
        3, 1, activation="sigmoid", dtype="float32", name="restored_candidate"
    )(x)
    unchanged_pixels = tf.keras.layers.Multiply(name="unchanged_pixels")(
        [image_input, inverse_mask]
    )
    filled_pixels = tf.keras.layers.Multiply(name="filled_pixels")(
        [restored_candidate, mask_input]
    )
    output = tf.keras.layers.Add(name="restored_image")(
        [unchanged_pixels, filled_pixels]
    )

    model = tf.keras.Model(
        {"hair_image": image_input, "hair_mask": mask_input},
        output,
        name="hair_removal_mobilenetv2_unet",
    )
    compile_model(model, config.learning_rate)
    return model


def _training_callbacks(
    model_path: str,
    metrics_csv_path: str,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    initial_best: float | None = None,
    append_csv: bool = False,
) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(metrics_csv_path, append=append_csv),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_masked_mae",
            mode="min",
            save_best_only=True,
            initial_value_threshold=initial_best,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_masked_mae",
            mode="min",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_masked_mae",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
    ]


def train_hair_removal_unet(
    clean_image_path: str = str(PROJECT_DIR / "dataset" / "hair"),
    hair_mask_path: str = str(PROJECT_DIR / "dataset" / "hair_mask"),
    config: HairRemovalUnetConfig | None = None,
) -> tf.keras.callbacks.History:
    config = config or HairRemovalUnetConfig()
    tf.keras.utils.set_random_seed(config.seed)
    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.metrics_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Tak samo jak w TrainMaskDetermiNet: zawsze utworz pierwszy zestaw przed
    # treningiem. Flaga --generate-once steruje tylko regeneracja miedzy epokami.
    generate_data(clean_image_path, hair_mask_path, config.seed)

    triplets = collect_training_triplets(hair_mask_path, clean_image_path)
    train_triplets, _ = split_triplets_by_source(
        triplets, config.validation_split, config.seed
    )
    train_source_ids = {source_image_id(item[0]) for item in train_triplets}
    train_dataset, validation_dataset = build_datasets(triplets, config)
    model = build_unet(config)

    def callbacks(
        initial_best: float | None = None, append_csv: bool = False
    ) -> list[tf.keras.callbacks.Callback]:
        result = _training_callbacks(
            config.model_path,
            config.metrics_csv_path,
            config.early_stopping_patience,
            config.early_stopping_min_delta,
            initial_best=initial_best,
            append_csv=append_csv,
        )
        if config.regenerate_each_epoch:
            result.insert(
                0,
                RegenerateTrainingData(
                    clean_image_path,
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

    completed_epochs = len(history.epoch)
    if completed_epochs < config.epochs:
        encoder = model.get_layer("mobilenetv2_encoder")
        encoder.trainable = True
        for index, layer in enumerate(encoder.layers):
            layer.trainable = index >= config.fine_tune_at and not isinstance(
                layer, tf.keras.layers.BatchNormalization
            )
        compile_model(model, config.fine_tune_learning_rate)

        baseline = float(
            model.evaluate(validation_dataset, verbose=0, return_dict=True)["masked_mae"]
        )
        fine_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            initial_epoch=completed_epochs,
            epochs=config.epochs,
            callbacks=callbacks(initial_best=baseline, append_csv=True),
        )
        for key, values in history.history.items():
            fine_history.history[key] = values + fine_history.history.get(key, [])
        fine_history.epoch = history.epoch + fine_history.epoch
        history = fine_history

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening U-Neta usuwajacego wlosy na podstawie obrazu i maski.")
    parser.add_argument("--epochs", type=int, default=HairRemovalUnetConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=HairRemovalUnetConfig.batch_size)
    parser.add_argument("--image-size", type=int, default=HairRemovalUnetConfig.image_size[0])
    parser.add_argument("--frozen-epochs", type=int, default=HairRemovalUnetConfig.frozen_epochs)
    parser.add_argument("--fine-tune-at", type=int, default=HairRemovalUnetConfig.fine_tune_at)
    parser.add_argument("--no-imagenet", action="store_true")
    parser.add_argument("--model-path", default=str(PROJECT_DIR / HairRemovalUnetConfig.model_path))
    parser.add_argument("--metrics-csv", default=str(PROJECT_DIR / HairRemovalUnetConfig.metrics_csv_path))
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=HairRemovalUnetConfig.early_stopping_patience,
        help="Zatrzymaj trening po tylu epokach bez istotnej poprawy val_masked_mae.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=HairRemovalUnetConfig.early_stopping_min_delta,
        help="Minimalny spadek val_masked_mae uznawany za poprawe.",
    )
    parser.add_argument("--generate-once", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_hair_removal_unet(
        config=HairRemovalUnetConfig(
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            epochs=args.epochs,
            frozen_epochs=args.frozen_epochs,
            fine_tune_at=args.fine_tune_at,
            encoder_weights=None if args.no_imagenet else "imagenet",
            model_path=args.model_path,
            metrics_csv_path=args.metrics_csv,
            regenerate_each_epoch=not args.generate_once,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
        ),
    )
