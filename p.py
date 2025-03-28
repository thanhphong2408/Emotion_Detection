import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Import OpenCV để xử lý ảnh

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.15,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_generator = train_datagen.flow_from_directory(
    'C:/Users/ADMIN/Downloads/ĐOANLHS/fer2013/train',
    target_size=(48, 48),
    batch_size=32,
    color_mode='rgb',  # Changed to RGB for ResNet
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'C:/Users/ADMIN/Downloads/ĐOANLHS/fer2013/validation',
    target_size=(48, 48),
    batch_size=32,
    color_mode='rgb',
    class_mode='categorical'
)

# Create model using transfer learning with ResNet50V2
base_model = ResNet50V2(
    weights='imagenet',
    include_top=False,
    input_shape=(48, 48, 3)
)

# Freeze base model layers
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# Compile with learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=0.00001
    )
]

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks
)

# Save the model
model.save('emotion_recognition_model_improved.h5')

# Plotting functions
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Enhanced prediction function with confidence scores
def predict_emotion(image_path):
    original_img = cv2.imread(image_path)  # Đọc ảnh gốc
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
    
    img = cv2.resize(original_img, (48, 48), interpolation=cv2.INTER_CUBIC)  # Resize với chất lượng cao
    img = img / 255.0  # Chuẩn hóa pixel về [0, 1]
    img_array = np.expand_dims(img, axis=0)  # Mở rộng chiều cho mô hình dự đoán
    
    prediction = model.predict(img_array)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    # Get top 3 predictions
    top_3_idx = prediction[0].argsort()[-3:][::-1]
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)  # Hiển thị ảnh gốc để không bị mờ
    plt.axis('off')
    plt.title('Input Image')
    
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(3), [prediction[0][i] * 100 for i in top_3_idx])
    plt.xticks(range(3), [emotion_labels[i] for i in top_3_idx], rotation=45)
    plt.ylabel('Confidence (%)')
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)

# Test the model
predict_emotion('fer2013/validation/angry/angry1.jpg')
