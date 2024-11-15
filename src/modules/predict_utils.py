from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model_path, img_path):
    try:
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please train the model first.")
        return None

    try:
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}. Ensure the file path is correct.")
        return None
    except Exception as e:
        print(f"Unexpected error while processing the image: {e}")
        return None

    try:
        prediction = model.predict(img_array)
        return "Smiling" if prediction > 0.5 else "Not Smiling"
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None