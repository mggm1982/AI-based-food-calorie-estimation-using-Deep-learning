"""
prediction.py - Food classification and calorie estimation pipeline
"""

import os
import io
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from PIL import Image


class FoodCaloriePredictor:
    def __init__(self, model_path=None, calorie_csv_path="resources/calorie_lookup.csv"):
        """
        Initialize the food calorie predictor
        Args:
            model_path: Path to saved model (if None, uses latest under models/saved_models)
            calorie_csv_path: Path to calorie lookup CSV
        """
        self.img_size = (224, 224)

        # Load model
        if model_path is None:
            model_path = self._find_latest_model()
        self.model = self._load_model(model_path)

        # Load class names
        self.class_names = self._load_class_names()

        # Load calorie database
        self.calorie_df = self._load_calorie_database(calorie_csv_path)

        print("✅ Food Calorie Predictor initialized successfully!")

    # --------- Model & metadata loaders ---------
    def _find_latest_model(self):
        """Find the latest saved model (prefers .h5; otherwise latest SavedModel dir)."""
        model_dir = Path("models/saved_models")
        if not model_dir.exists():
            raise FileNotFoundError("models/saved_models not found. Train a model first.")

        # Prefer .h5 files (often easiest to load)
        h5_files = list(model_dir.glob("*.h5"))
        if h5_files:
            latest_h5 = max(h5_files, key=os.path.getmtime)
            return str(latest_h5)

        # Fallback: SavedModel directories (skip obvious non-model folders)
        candidates = [d for d in model_dir.iterdir() if d.is_dir()]
        if not candidates:
            raise FileNotFoundError("No trained model found in models/saved_models.")
        latest_model = max(candidates, key=os.path.getmtime)
        return str(latest_model)

    def _load_model(self, model_path):
        """Load the trained model (SavedModel dir or .h5)."""
        print(f"📁 Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded: {getattr(model, 'name', 'keras_model')}")
        return model

    def _load_class_names(self):
        """Load class names from JSON file"""
        class_names_path = Path("resources/class_names.json")
        if class_names_path.exists():
            with open(class_names_path, "r") as f:
                class_names = json.load(f)
            print(f"✅ Loaded {len(class_names)} class names from resources/class_names.json")
            return class_names

        # Sensible fallback list (will likely NOT match Food-101 fully)
        fallback = [
            "pizza", "hamburger", "sushi", "ramen", "ice_cream",
            "french_fries", "donuts", "cake", "sandwich", "pasta"
        ]
        print("⚠️ resources/class_names.json not found. Using fallback class names.")
        return fallback

    def _load_calorie_database(self, csv_path):
        """Load calorie information from CSV or create a default one"""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print("⚠️ Calorie CSV not found. Creating default database...")
            self._create_default_calorie_db(csv_path)

        df = pd.read_csv(csv_path)
        # Normalize food_name column to underscore style for matching
        if "food_name" in df.columns:
            df["food_name"] = df["food_name"].astype(str)
        print(f"✅ Loaded calorie data for {len(df)} items from {csv_path}")
        return df

    def _create_default_calorie_db(self, csv_path):
        """Create a default calorie database"""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        default_data = {
            "food_name": [
                "pizza", "hamburger", "sushi", "ramen", "ice_cream",
                "french_fries", "donuts", "cake", "sandwich", "pasta",
                "salad", "steak", "chicken_wings", "tacos", "waffles",
                "pancakes", "bread", "soup", "rice", "omelette"
            ],
            "serving_size": [
                "1 slice", "1 burger", "8 pieces", "1 bowl", "1 cup",
                "1 medium", "1 donut", "1 slice", "1 sandwich", "1 cup",
                "1 bowl", "1 piece", "6 pieces", "2 tacos", "2 waffles",
                "3 pancakes", "2 slices", "1 bowl", "1 cup", "1 large"
            ],
            "calories": [
                285, 354, 280, 436, 273,
                365, 253, 257, 256, 220,
                152, 542, 441, 410, 290,
                320, 160, 120, 206, 154
            ],
            "category": [
                "Fast Food", "Fast Food", "Japanese", "Japanese", "Dessert",
                "Fast Food", "Dessert", "Dessert", "Lunch", "Italian",
                "Healthy", "Protein", "Protein", "Mexican", "Breakfast",
                "Breakfast", "Bakery", "Healthy", "Staple", "Breakfast"
            ],
        }
        pd.DataFrame(default_data).to_csv(csv_path, index=False)
        print(f"✅ Created default calorie database at: {csv_path}")

    # --------- Inference helpers ---------
    def preprocess_image(self, image_input):
        """Preprocess image for model input. Accepts path, PIL.Image, NumPy array, or bytes."""
        # 1) Load as PIL.Image
        if isinstance(image_input, (str, Path)):
            img = Image.open(str(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        elif isinstance(image_input, (bytes, bytearray, io.BytesIO)):
            if isinstance(image_input, io.BytesIO):
                img = Image.open(image_input)
            else:
                img = Image.open(io.BytesIO(image_input))
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 2) Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # 3) Resize & preprocess for MobileNetV2
        img = img.resize(self.img_size, Image.Resampling.LANCZOS)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array

    def predict_food(self, image_input, top_k=3):
        """
        Predict food type from image
        Args:
            image_input: Path, PIL Image, NumPy array, or bytes
        Returns:
            List of dicts: [{'food': name, 'confidence': float}, ...]
        """
        x = self.preprocess_image(image_input)
        preds = self.model.predict(x, verbose=0)

        # Safety: if model outputs float16 (mixed precision), cast to float32
        preds = np.asarray(preds, dtype=np.float32)

        # Top-K
        top_idx = np.argsort(preds[0])[-top_k:][::-1]
        results = []
        for idx in top_idx:
            # Guard against mismatched lengths
            if idx < len(self.class_names):
                food_name = self.class_names[idx]
            else:
                food_name = f"class_{idx}"
            confidence = float(preds[0][idx])
            results.append({"food": food_name, "confidence": confidence})
        return results

    def get_calorie_info(self, food_name):
        """
        Look up calories/macros for a predicted food (underscore style).
        Tries exact match, then loose contains() on the first token.
        """
        # Exact match first
        df = self.calorie_df
        exact = df[df["food_name"].str.lower() == str(food_name).lower()]
        if not exact.empty:
            row = exact.iloc[0]
        else:
            token = str(food_name).split("_")[0]
            fuzzy = df[df["food_name"].str.contains(token, case=False, na=False)]
            row = fuzzy.iloc[0] if not fuzzy.empty else None

        if row is not None:
            return {
                "food_name": row.get("food_name", food_name),
                "serving_size": row.get("serving_size", "standard serving"),
                "calories": int(row.get("calories", 200)),
                "protein_g": float(row.get("protein_g", 0)),
                "carbs_g": float(row.get("carbs_g", 0)),
                "fat_g": float(row.get("fat_g", 0)),
                "category": row.get("category", "General"),
            }

        # Default if not found
        return {
            "food_name": food_name,
            "serving_size": "standard serving",
            "calories": 200,
            "protein_g": 10.0,
            "carbs_g": 25.0,
            "fat_g": 8.0,
            "category": "Unknown",
        }

    def predict_calories(self, image_input, confidence_threshold=0.30):
        """
        Complete pipeline: predict food and estimate calories
        """
        predictions = self.predict_food(image_input, top_k=3)
        top = predictions[0]

        if top["confidence"] < confidence_threshold:
            return {
                "success": False,
                "message": f"Low confidence ({top['confidence']:.1%}). Unable to identify food accurately.",
                "predictions": predictions,
            }

        calorie_info = self.get_calorie_info(top["food"])
        result = {
            "success": True,
            "food_name": calorie_info["food_name"],
            "confidence": float(top["confidence"]),
            "serving_size": calorie_info["serving_size"],
            "calories": int(calorie_info["calories"]),
            "protein_g": float(calorie_info["protein_g"]),
            "carbs_g": float(calorie_info["carbs_g"]),
            "fat_g": float(calorie_info["fat_g"]),
            "category": calorie_info["category"],
            "alternative_predictions": predictions[1:3],
        }
        return result

    # --------- CLI pretty print ---------
    def print_prediction(self, result):
        if result["success"]:
            print("\n" + "=" * 50)
            print("🍔 FOOD DETECTION RESULTS 🍔")
            print("=" * 50)
            print(f"🎯 Food Detected: {result['food_name'].upper()}")
            print(f"📊 Confidence: {result['confidence']:.1%}")
            print(f"🍽️ Serving Size: {result['serving_size']}")
            print(f"🔥 Calories: {result['calories']} kcal")
            print(f"🥩 Protein: {result['protein_g']} g")
            print(f"🍞 Carbs: {result['carbs_g']} g")
            print(f"🧈 Fat: {result['fat_g']} g")
            print(f"📁 Category: {result['category']}")
            if result["alternative_predictions"]:
                print("\n📋 Other possibilities:")
                for alt in result["alternative_predictions"]:
                    print(f"   - {alt['food']}: {alt['confidence']:.1%}")
        else:
            print("\n⚠️ " + result["message"])
            print("\n📋 Top predictions:")
            for pred in result["predictions"]:
                print(f"   - {pred['food']}: {pred['confidence']:.1%}")


# --------- Quick test / CLI ---------
def test_prediction_pipeline():
    print("🧪 Testing Food Calorie Prediction Pipeline\n")
    predictor = FoodCaloriePredictor()

    test_dir = Path("data/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Dummy test image (replace with real images in data/test_images/)
    test_img = Image.new("RGB", (224, 224), color=(255, 200, 100))
    test_img_path = test_dir / "test_food.jpg"
    test_img.save(test_img_path)

    result = predictor.predict_calories(test_img_path)
    predictor.print_prediction(result)
    print("\n💡 TIP: Place your own food images in 'data/test_images/' then run:\n"
          "python -m src.prediction data/test_images/your_food.jpg")
    return predictor


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"❌ Error: Image file '{image_path}' not found!")
            raise SystemExit(1)
        print(f"🖼️ Processing image: {image_path}\n")
        predictor = FoodCaloriePredictor()
        result = predictor.predict_calories(image_path)
        predictor.print_prediction(result)
    else:
        test_prediction_pipeline()
