import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import pickle
import os
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))


# Завантаження моделей
def load_models():
    cnn_model = tf.keras.models.load_model(os.path.join(current_directory, 'fashion_mnist_cnn_model.h5'))
    vgg16_model = tf.keras.models.load_model(os.path.join(current_directory, 'fashion_mnist_fine_tuned_vgg16.h5'))
    return cnn_model, vgg16_model


# Завантаження зображення та передобробка для CNN
def load_image(img_file):
    img = image.load_img(img_file, target_size=(28, 28),
                         color_mode='grayscale')  # Завантаження зображення Fashion MNIST
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Нормалізація для CNN моделі
    return img_array


# Завантаження зображення та передобробка для VGG16
def load_image_vgg(img_file):
    img = image.load_img(img_file, target_size=(32, 32), color_mode='grayscale')  # Перетворення у 224x224
    img_array = image.img_to_array(img)
    img_array = np.repeat(img_array, 3, axis=-1)  # Перетворення зображення в 3 канали
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Попередня обробка для VGG16
    return img_array


# Функція прогнозування
def predict(model, img_array):
    prediction = model.predict(img_array)
    return prediction


# Завантаження історії з файлу
def load_history(history_file):
    if os.path.exists(history_file):
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
            return history
    else:
        st.write(f"Файл історії не знайдений за шляхом: {history_file}")
        return None


# Візуалізація результатів
def plot_loss_accuracy(history):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Графік функції втрат
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Over Epochs')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Графік точності
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy Over Epochs')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        st.pyplot(fig)

    except Exception as e:
        st.write(f"Помилка при побудові графіків: {e}")
        return


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # для можливості відоюраження графіків

    st.title("Застосунок для класифікації зображень Fashion MNIST")

    # Випадаючий список для вибору моделі за якою робити прогнозування
    model_choice = st.selectbox("Вибір моделі", ("CNN", "VGG16"))

    # Завантаження зображення
    img_file = st.file_uploader("Завантажте зображення", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        # Передобробка зображення
        if model_choice == "VGG16":
            img_array = load_image_vgg(img_file)
        else:
            img_array = load_image(img_file)

        # Завантаження моделей
        cnn_model, vgg16_model = load_models()

        # Завантажкення моделі і її історії після вибору в списку
        if model_choice == "CNN":
            model = cnn_model
            history_file = os.path.join(current_directory, 'cnn_model_history_new.pickle')
        else:
            model = vgg16_model
            history_file = os.path.join(current_directory, 'fashion_mnist_fine_tuned_vgg16_history.pickle')

            # Прогнозування
        prediction = predict(model, img_array)

        st.image(img_file, caption="Вхідне зображення", use_container_width=False, width=200)

        st.write(f"Результати класифікації:")

        # Виведення ймовірностей для кожного класу з форматуванням
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]  # Список класів Fashion MNIST

        # Форматований вивід ймовірностей
        probabilities = prediction[0]

        # Визначаємо передбачений клас
        predicted_class = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class]
        st.write(f"**Передбачений клас: {predicted_class_name}**")

        st.write("Ймовірності для кожного класу:")
        # Створюємо таблицю для ймовірностей
        probability_table = {class_names[i]: f"{probabilities[i]:.4f}" for i in range(len(class_names))}
        df = pd.DataFrame(list(probability_table.items()), columns=["Клас", "Ймовірність"])
        st.table(df.set_index("Клас", drop=True))  # Видаляємо індекс

        # Завантаження історії тренування для вибраної моделі
        history = load_history(history_file)

        # Графіки функції втрат і точності для вибраної моделі
        if history:
            plot_loss_accuracy(history)
        else:
            st.write("Не вдалося завантажити історію тренування.")


# Запуск додатку
if __name__ == "__main__":
    main()