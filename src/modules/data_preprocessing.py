import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_smiling_data(file_path, data_dir, sample_size=20000):
    # Load data
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, header=None)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Attribute file not found at {file_path}. Ensure the file is downloaded and located correctly.") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while loading data from {file_path}: {e}") from e

    headers = ['image_id', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
               'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
               'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
               'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
               'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
               'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
               'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
               'Wearing_Necktie', 'Young']
    df.columns = headers

    # Prepare smiling data
    df_smiling = df[['image_id', 'Smiling']]
    df_smiling['Smiling'] = df_smiling['Smiling'].replace(-1, 0).astype(str)

    
    # Data generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_dataframe(
        dataframe=df_smiling,
        directory=data_dir,
        x_col='image_id',
        y_col='Smiling',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=df_smiling,
        directory=data_dir,
        x_col='image_id',
        y_col='Smiling',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator