import tensorflow as tf

def evaluate_model(model_path, validation_generator):
    # Load the best model
    best_model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    val_loss, val_acc = best_model.evaluate(validation_generator)
    print(f'Validation Accuracy: {val_acc}')
    return val_loss, val_acc