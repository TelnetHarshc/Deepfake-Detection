# Convolutional Neural Network (CNN) built with the Keras Sequential API
# import tensorflow as tf
# from tensorflow.keras import layers, models
# def build_model():
#     model = models.Sequential([
#         layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
#         layers.MaxPooling2D(2,2),

#         layers.Conv2D(64, (3,3), activation='relu'),
#         layers.MaxPooling2D(2,2),

#         layers.Conv2D(128, (3,3), activation='relu'),
#         layers.MaxPooling2D(2,2),

#         layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='sigmoid')  # Binary classification
#     ])

#     model.compile(
#         optimizer='adamW',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     return model


#vgg16
# import tensorflow as tf
# from tensorflow.keras import layers, models

# def build_model():
#     base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     base_model.trainable = False  # Freeze base layers

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(512, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='sigmoid')
#     ])

#     # Use a float learning rate instead
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
#                   loss='binary_crossentropy', metrics=['accuracy'])

#     return model  # Corrected return statement



## Resnet50

# import tensorflow as tf
# from tensorflow.keras import layers, models

# def build_model(base="ResNet50"):
#     if base == "ResNet50":
#         base_model = tf.keras.applications.ResNet50(
#             weights='imagenet', include_top=False, input_shape=(224, 224, 3)
#         )
#     elif base == "ResNet152":
#         base_model = tf.keras.applications.ResNet152(
#             weights='imagenet', include_top=False, input_shape=(224, 224, 3)
#         )
#     else:
#         raise ValueError("Invalid base model. Choose 'ResNet50' or 'ResNet152'.")

#     base_model.trainable = False  # Freeze initial layers

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(512, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='sigmoid')  # Binary classification
#     ])

#     # Compile Model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy', metrics=['accuracy'])

#     return model



#Vgg16 + Lstm
# import tensorflow as tf
# from tensorflow.keras import layers, models

# def build_model():
#     base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     base_model.trainable = False  # Freeze base layers

#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(512, activation='relu'),
#         layers.BatchNormalization(),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation='sigmoid')  # Binary classification
#     ])

#     # Compile Model
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#                   loss='binary_crossentropy', metrics=['accuracy'])

#     return model

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    base_model = tf.keras.applications.VGG16(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )

    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze last 4 layers
    for layer in base_model.layers[-4:]:
        layer.trainable = True

    # Build the full model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # smaller LR for fine-tuning
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
