# import tensorflow as tf
# from tensorflow.keras import layers

# def load_datasets(train_dir="Dataset/Train", val_dir="Dataset/Validation", test_dir="Dataset/Test", img_size=(224, 224), batch_size=32):
#     # Load datasets
#     train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=img_size, batch_size=batch_size)
#     val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, image_size=img_size, batch_size=batch_size)
#     test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, image_size=img_size, batch_size=batch_size)

#     # Data Augmentation
#     data_augmentation = tf.keras.Sequential([
#         layers.RandomFlip("horizontal_and_vertical"),
#         layers.RandomRotation(0.2),
#         layers.RandomZoom(0.2),
#         layers.RandomContrast(0.2),
#     ])
#     train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

#     # Normalize pixel values
#     normalization_layer = layers.Rescaling(1./255)
#     train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#     val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
#     test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#     return train_ds, val_ds, test_ds

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_datasets(train_dir="Dataset/Train", val_dir="Dataset/Validation", test_dir="Dataset/Test", img_size=(224, 224), batch_size=32):
    # 1. Train data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )

    # 2. Validation and test data (no augmentation, just rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    # 3. Load the datasets from directory
    train_ds = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_ds = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    test_ds = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_ds, val_ds, test_ds
