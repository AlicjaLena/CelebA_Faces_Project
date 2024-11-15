from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def get_callbacks(checkpoint_path='best_model.keras', monitor='val_loss', patience=3, verbose=1):
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path, 
        save_best_only=True, 
        monitor=monitor, 
        mode='min'
    )
    early_stopping = EarlyStopping(
        monitor=monitor, 
        patience=patience, 
        restore_best_weights=True,
        verbose = verbose
    )
    return [checkpoint, early_stopping]