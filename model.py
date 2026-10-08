import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.optimizers import Adam

def create_convnext_model(input_shape=(224, 224, 3), num_classes=2, freeze_backbone=True):
    """
    Create a ConvNeXt-Tiny based model for violence detection.

    Args:
        input_shape: Tuple, shape of input images (height, width, channels)
        num_classes: Int, number of output classes
        freeze_backbone: Boolean, whether to freeze the ConvNeXt backbone

    Returns:
        A compiled Keras model
    """
    # Load pre-trained ConvNeXt-Tiny model from Keras applications
    base_model = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'  # Use global average pooling
    )

    # Freeze the backbone if specified
    if freeze_backbone:
        for layer in base_model.layers:
            layer.trainable = False

    # Create the model
    inputs = tf.keras.Input(shape=input_shape)

    # Preprocess input for ConvNeXt (ImageNet normalization)
    x = tf.keras.applications.convnext.preprocess_input(inputs)

    # Get features from backbone
    features = base_model(x, training=not freeze_backbone)

    # Add dropout for regularization
    x = layers.Dropout(0.5)(features)

    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model

def compile_model(model, learning_rate=1e-4):
    """
    Compile the model with appropriate loss and metrics.
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_data_augmentation():
    """
    Create data augmentation layer for training.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.2),
    ])

def preprocess_image(image, label=None, target_size=(224, 224)):
    """
    Preprocess image for ConvNeXt model.
    """
    # Resize to target size
    image = tf.image.resize(image, target_size)

    # Convert grayscale to RGB if needed
    if image.shape[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    elif image.shape[-1] == 4:  # If RGBA, convert to RGB
        image = image[..., :3]

    # Ensure 3 channels
    image = tf.ensure_shape(image, (*target_size, 3))

    if label is not None:
        return image, label
    return image
