"""
data_preparation.py — Food-101 loader using official splits (no copying needed)
Builds tf.data pipelines that match MobileNetV2 preprocessing.
"""

from pathlib import Path
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

class Food101Preparator:
    """
    Loads Food-101 from:
        data/raw/food-101/
            ├─ images/<class>/<image>.jpg
            └─ meta/{classes.txt, train.txt, test.txt}
    Returns tf.data.Dataset objects for train/val with consistent preprocessing.
    """

    def __init__(
        self,
        root_dir: str = "data/raw/food-101",
        img_size: tuple[int, int] = (224, 224),
        num_classes: int | None = 20,     # set None for all 101 classes
        per_class_limit: int | None = None  # cap images per class (speed-up)
    ):
        self.root = Path(root_dir)
        self.images_dir = self.root / "images"
        self.meta_dir = self.root / "meta"
        self.img_size = img_size

        if not self.images_dir.exists() or not self.meta_dir.exists():
            raise FileNotFoundError(
                f"Expected {self.root}/images and {self.root}/meta to exist."
            )

        # Read class list (official Food-101)
        self.class_names_all = (self.meta_dir / "classes.txt").read_text().strip().splitlines()

        # Subset classes if requested (preserves official class order)
        if num_classes is not None:
            self.class_names = self.class_names_all[:num_classes]
        else:
            self.class_names = self.class_names_all

        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        # Read official splits. Each line is like "pizza/1000001"
        train_ids_all = (self.meta_dir / "train.txt").read_text().strip().splitlines()
        test_ids_all  = (self.meta_dir / "test.txt").read_text().strip().splitlines()

        # Filter to selected classes only
        def keep(line: str) -> bool:
            return line.split("/")[0] in self.class_to_idx

        self.train_ids = [x for x in train_ids_all if keep(x)]
        self.val_ids   = [x for x in test_ids_all  if keep(x)]

        # Optionally cap per-class counts (useful on CPU or small GPU)
        if per_class_limit is not None:
            self.train_ids = self._limit_per_class(self.train_ids, per_class_limit)
            self.val_ids   = self._limit_per_class(self.val_ids, per_class_limit)

    def _limit_per_class(self, id_list: list[str], limit: int) -> list[str]:
        counts = {c: 0 for c in self.class_names}
        kept = []
        for item in id_list:
            cls = item.split("/")[0]
            if counts[cls] < limit:
                kept.append(item)
                counts[cls] += 1
        return kept

    def _paths_and_labels(self, id_list: list[str]):
        paths = [str(self.images_dir / f"{lid}.jpg") for lid in id_list]
        labels = [self.class_to_idx[lid.split('/')[0]] for lid in id_list]
        return paths, labels

    @tf.function
    def _load_and_preprocess(self, path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_size)
        # IMPORTANT: same preprocessing used in your training/inference (MobileNetV2)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        label = tf.one_hot(label, depth=len(self.class_names))
        return img, label

    def _augment(self, img, label):
        # Light, safe augmentations
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    def prepare_tensorflow_dataset(self, batch_size: int = 32, augment: bool = True):
        train_paths, train_labels = self._paths_and_labels(self.train_ids)
        val_paths,   val_labels   = self._paths_and_labels(self.val_ids)

        train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        val_ds   = tf.data.Dataset.from_tensor_slices((val_paths,   val_labels))

        train_ds = (train_ds
                    .shuffle(buffer_size=max(len(train_paths), 1000))
                    .map(self._load_and_preprocess, num_parallel_calls=AUTOTUNE))
        if augment:
            train_ds = train_ds.map(self._augment, num_parallel_calls=AUTOTUNE)

        val_ds = val_ds.map(self._load_and_preprocess, num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
        val_ds   = val_ds.batch(batch_size).prefetch(AUTOTUNE)

        return train_ds, val_ds, self.class_names


# Optional CLI: quick sanity check
if __name__ == "__main__":
    prep = Food101Preparator(root_dir="data/raw/food-101", num_classes=20)
    tr, va, names = prep.prepare_tensorflow_dataset(batch_size=32)
    print(f"Classes ({len(names)}): {names[:10]}...")
    print(f"Train batches: {len(tr)}, Val batches: {len(va)}")
