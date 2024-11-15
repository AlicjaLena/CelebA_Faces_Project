import tensorflow as tf

# Ścieżka do zapisanej wersji modelu
model_path = 'best_model.keras'

# Wczytanie modelu
best_model = tf.keras.models.load_model(model_path)

# Wyświetlenie podsumowania modelu (opcjonalnie)
best_model.summary()
