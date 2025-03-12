import tensorflow as tf

# Path to your SavedModel directory
saved_model_path = "working_model_1"
output_h5_path = "converted_model.h5"

try:
    print("🔹 Loading SavedModel...")
    model = tf.keras.models.load_model(saved_model_path)  # Attempt to load as a Keras model
    
    print("✅ SavedModel loaded successfully!")
    print("🔹 Converting to H5 format...")

    model.save(output_h5_path, save_format="h5")

    print(f"✅ Model converted and saved as '{output_h5_path}'")
except Exception as e:
    print(f"❌ Failed to convert: {e}")
