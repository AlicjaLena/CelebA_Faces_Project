import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from modules.data_preprocessing import prepare_smiling_data
from modules.model_builder import build_model
from modules.evaluation import evaluate_model
from modules.visualization import plot_training_history
from modules.predict_utils import predict_image

# src/ path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_PATH = 'data/list_attr_celeba.txt'
IMAGES_DIR = r'C:\Users\Ala\tensorflow_datasets\img_align_celeba'
MODEL_PATH = 'best_model.keras'
SAMPLE_IMAGE_PATH = 'data/samples/sample_image_1.jpg'

def main(): 
    # Prepare data with 10,000 samples
    train_generator, validation_generator = prepare_smiling_data(DATA_PATH, IMAGES_DIR, sample_size=20000)

    # Build the model
    model = build_model()

    # Callbacks
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model and plot the history
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=50,
        callbacks=[checkpoint, early_stopping]
    )

    # Visualize the results
    plot_training_history(history)

    # Evaluate the model
    try:
        best_model = tf.keras.models.load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
        return
    
    best_model.summary()
    evaluate_model(MODEL_PATH, validation_generator)

    # Predict on a sample image 
    result = predict_image(MODEL_PATH, SAMPLE_IMAGE_PATH)
    print(f'Prediction: {result}')
    

if __name__ == "__main__":
    main()
