import tensorflow as tf
from keras import layers, models
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras.models

# Define emotion labels
EMOTION_LABELS = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

def create_emotion_cnn():
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15), 

        # Second Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15), 

        # Third Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15), 

        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(7, activation='softmax')
    ])
    return model

def load_and_preprocess_data(train_dir, test_dir):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS.keys()}

    print(f"Looking for training images in: {os.path.abspath(train_dir)}")
    print(f"Looking for test images in: {os.path.abspath(test_dir)}")

    # Load training data
    for emotion, label in EMOTION_LABELS.items():
        emotion_dir = os.path.join(train_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Training directory not found: {emotion_dir}")
            continue

        print(f"Processing training directory: {emotion_dir}")
        for img_name in os.listdir(emotion_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(emotion_dir, img_name)
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        img_path,
                        color_mode='grayscale',
                        target_size=(48, 48)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize

                    train_images.append(img_array)
                    train_labels.append(label)
                    emotion_counts[emotion] += 1
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")

    # Load test data
    for emotion, label in EMOTION_LABELS.items():
        emotion_dir = os.path.join(test_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Test directory not found: {emotion_dir}")
            continue

        print(f"Processing test directory: {emotion_dir}")
        for img_name in os.listdir(emotion_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(emotion_dir, img_name)
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        img_path,
                        color_mode='grayscale',
                        target_size=(48, 48)
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize

                    test_images.append(img_array)
                    test_labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")

    print("\nEmotion distribution in training set:")
    for emotion, count in emotion_counts.items():
        print(f"    {emotion}: {count} images")

    print(f"\nTotal training images loaded: {len(train_images)}")
    print(f"Total test images loaded: {len(test_images)}")

    if len(train_images) == 0 or len(test_images) == 0:
        raise ValueError("No images were loaded from either training or test set!")

    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    X_test = np.array(test_images)
    y_test = np.array(test_labels)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test

def create_data_generators(X_tr, y_tr, X_val, y_val):
    # Augmentări blânde: rotire ±10°, translatare ±10%, flip
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = train_datagen.flow(
        X_tr,
        tf.keras.utils.to_categorical(y_tr, num_classes=7),
        batch_size=128,
        shuffle=True
    )

    # Pentru validare: doar normalizare (fără augmentări)
    val_datagen = ImageDataGenerator()
    val_gen = val_datagen.flow(
        X_val,
        tf.keras.utils.to_categorical(y_val, num_classes=7),
        batch_size=128,
        shuffle=False
    )

    print("\nTraining set class distribution:")
    tr_counts = np.bincount(y_tr, minlength=7)
    for idx, emotion in enumerate(EMOTION_LABELS.keys()):
        print(f"    {emotion}: {tr_counts[idx]} samples")

    print("\nValidation set class distribution:")
    val_counts = np.bincount(y_val, minlength=7)
    for idx, emotion in enumerate(EMOTION_LABELS.keys()):
        print(f"    {emotion}: {val_counts[idx]} samples")

    return train_gen, val_gen

def compute_class_weights(y_train):
    # Compute class weights to address imbalance
    cw_array = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: float(cw_array[i]) for i in range(len(cw_array))}
    print("\nComputed class weights:")
    for idx, emotion in enumerate(EMOTION_LABELS.keys()):
        print(f"    {emotion}: {class_weights[idx]:.2f}")
    return class_weights

def train_model(model, train_gen, val_gen, class_weights):
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3, 
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]

    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history

def evaluate_on_test(model, X_test, y_test):
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=7)
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
    print(f"\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Directories for your data
    train_dir = 'input/facial/train'
    test_dir = 'input/facial/test'
    print(f"Current working directory: {os.getcwd()}")

    # Load data
    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_dir, test_dir)

    # Split train into train/validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # Create data generators
    train_gen, val_gen = create_data_generators(X_tr, y_tr, X_val, y_val)

    # Compute class weights
    class_weights = compute_class_weights(y_train)

    # Create and train model
    model = create_emotion_cnn()
    history = train_model(model, train_gen, val_gen, class_weights)

    # Plot training history
    plot_training_history(history)

    # Load best saved model and evaluate on test
    best_model = tf.keras.models.load_model('models_facial/best_model.keras')
    evaluate_on_test(best_model, X_test, y_test)

    # Save final model
    best_model.save('models_facial/facial_emotion_model_final.keras')

    # Print class mapping
    print("\nEmotion Class Mapping:")
    for emotion, idx in EMOTION_LABELS.items():
        print(f"    {emotion}: {idx}")


if __name__ == "__main__":
    main()
