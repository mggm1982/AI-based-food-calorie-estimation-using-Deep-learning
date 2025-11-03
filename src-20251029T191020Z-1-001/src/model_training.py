"""
model_training.py - Train food classification model using transfer learning
(compatible with Food-101 official splits and Streamlit inference)
"""

import os
import json
from pathlib import Path
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ✅ Import the Food-101 preparator
from src.data_preparation import Food101Preparator

# ---- Optional: enable mixed precision on GPU for speed ----
try:
    if tf.config.list_physical_devices("GPU"):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        USING_MP = True
        print("⚡ Mixed precision enabled (GPU detected).")
    else:
        USING_MP = False
        print("CPU training: mixed precision not enabled.")
except Exception:
    USING_MP = False

class FoodClassifierTrainer:
    def __init__(self, model_name="MobileNetV2", num_classes=20):
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = (224, 224)
        self.model = None
        self.base_model = None
        self.history = None

        self.model_dir = Path("models/saved_models")
        self.checkpoint_dir = Path("models/checkpoints")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_model(self):
        """Create transfer-learning model (MobileNetV2/EfficientNetB0) with correct preprocessing."""
        print(f"\n🏗️ Building {self.model_name} model...")

        inputs = keras.Input(shape=(*self.img_size, 3))
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)

        if self.model_name == "MobileNetV2":
            base_model = MobileNetV2(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights="imagenet"
            )
        elif self.model_name == "EfficientNetB0":
            base_model = EfficientNetB0(
                input_shape=(*self.img_size, 3),
                include_top=False,
                weights="imagenet"
            )
        else:
            raise ValueError("model_name must be 'MobileNetV2' or 'EfficientNetB0'.")

        base_model.trainable = False
        x = base_model(x, training=False)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # IMPORTANT: keep final logits in float32 if using mixed precision
        out_dtype = "float32" if USING_MP else None
        outputs = layers.Dense(self.num_classes, activation="softmax", dtype=out_dtype)(x)

        self.model = keras.Model(inputs, outputs, name=f"{self.model_name}_food101")
        self.base_model = base_model

        print(f"✅ Model created with {self.model.count_params():,} parameters")
        return self.model

    def compile_model(self, learning_rate=1e-3):
        print(f"\n⚙️ Compiling model with lr={learning_rate}")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
        )
        print("✅ Model compiled.")

    def train_initial(self, train_ds, val_ds, epochs=10):
        print("\n🎯 Phase 1: Train with frozen base.")
        callbacks = self._get_callbacks("initial")
        self.history = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1
        )
        print("✅ Initial training complete.")
        return self.history

    def fine_tune(self, train_ds, val_ds, epochs=10, unfreeze_layers=30):
        print(f"\n🎯 Phase 2: Fine-tune last {unfreeze_layers} layers.")
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        self.compile_model(learning_rate=1e-4)

        callbacks = self._get_callbacks("finetune")
        hist_ft = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=self.history.epoch[-1] + 1 if self.history else 0
        )

        if self.history:
            for k, v in hist_ft.history.items():
                self.history.history.setdefault(k, [])
                self.history.history[k].extend(v)
        else:
            self.history = hist_ft

        print("✅ Fine-tuning complete.")
        return self.history

    def _get_callbacks(self, phase="initial"):
        cbs = []
        ckpt_path = self.checkpoint_dir / f"best_{self.model_name}_{phase}.h5"
        cbs.append(ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ))
        cbs.append(EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ))
        cbs.append(ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ))
        return cbs

    def save_model(self, model_name=None):
        if model_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"food_classifier_{self.model_name}_{ts}"

        savedmodel_dir = self.model_dir / model_name
        self.model.save(savedmodel_dir)
        print(f"✅ SavedModel: {savedmodel_dir}")

        h5_path = self.model_dir / f"{model_name}.h5"
        self.model.save(h5_path)
        print(f"✅ H5: {h5_path}")

        # TFLite (best-effort)
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            tflite_bytes = converter.convert()
            tflite_path = self.model_dir / f"{model_name}.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_bytes)
            print(f"✅ TFLite: {tflite_path}")
        except Exception as e:
            print(f"⚠️ TFLite conversion skipped: {e}")

        return str(savedmodel_dir)

    def plot_training_history(self):
        if not self.history:
            print("⚠️ No history to plot.")
            return

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self.history.history["accuracy"], label="train_acc")
        axes[0].plot(self.history.history["val_accuracy"], label="val_acc")
        axes[0].set_title("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history.history["loss"], label="train_loss")
        axes[1].plot(self.history.history["val_loss"], label="val_loss")
        axes[1].set_title("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        Path("models").mkdir(exist_ok=True)
        out = Path("models/training_history.png")
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        print(f"📊 Training history saved to {out}")

    def evaluate_model(self, test_ds):
        print("\n📈 Evaluating…")
        results = self.model.evaluate(test_ds, verbose=1)
        for name, val in zip(self.model.metrics_names, results):
            print(f"{name}: {val:.4f}")
        return results


def train_food_classifier():
    print("🍔 Food Classifier Training Pipeline 🍔\n")

    # ---- STEP 1: Prepare data (Food-101) ----
    print("="*50)
    print("STEP 1: Preparing Dataset (official splits)")
    print("="*50)

    preparator = Food101Preparator(
        root_dir="data/raw/food-101",
        num_classes=20,          # increase to None for all 101 classes
        per_class_limit=None     # set (e.g., 400) to speed up experiments
    )

    train_ds, val_ds, class_names = preparator.prepare_tensorflow_dataset(
        batch_size=32,  # tune as per VRAM
        augment=True
    )

    Path("resources").mkdir(exist_ok=True)
    with open("resources/class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"✅ Saved {len(class_names)} class names to resources/class_names.json")

    # ---- STEP 2: Create & compile model ----
    print("\n" + "="*50)
    print("STEP 2: Creating Model")
    print("="*50)

    trainer = FoodClassifierTrainer(
        model_name="MobileNetV2",
        num_classes=len(class_names)
    )
    trainer.create_model()
    trainer.compile_model(learning_rate=1e-3)

    # ---- STEP 3: Train (frozen base) ----
    print("\n" + "="*50)
    print("STEP 3: Training (Phase 1)")
    print("="*50)
    trainer.train_initial(train_ds, val_ds, epochs=10)

    # ---- STEP 4: Fine-tune ----
    print("\n" + "="*50)
    print("STEP 4: Fine-tuning (Phase 2)")
    print("="*50)
    # Unfreeze more layers for better accuracy; adjust for GPU memory
    trainer.fine_tune(train_ds, val_ds, epochs=10, unfreeze_layers=30)

    # ---- STEP 5: Save model ----
    print("\n" + "="*50)
    print("STEP 5: Saving Model")
    print("="*50)
    model_path = trainer.save_model("food_classifier_model")

    # ---- STEP 6: Plot & Evaluate ----
    print("\n" + "="*50)
    print("STEP 6: Report & Evaluation")
    print("="*50)
    trainer.plot_training_history()
    trainer.evaluate_model(val_ds)

    print("\n" + "="*50)
    print("🎉 TRAINING COMPLETE! 🎉")
    print("="*50)
    print(f"✅ Model saved to: {model_path}")
    print("✅ Class names saved to: resources/class_names.json")
    print("✅ Training history saved to: models/training_history.png")

    return model_path, class_names


if __name__ == "__main__":
    train_food_classifier()
