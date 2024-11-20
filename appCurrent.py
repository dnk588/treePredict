import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import json
import folium
from streamlit_folium import st_folium

# Загрузка модели
model = load_model('trained_model_breed_current.keras')
st.title("Прогнозирование состояния дерева")

# Классификации
# Загрузка данных из JSON-файлов
with open('tree-breeds.json', 'r', encoding='utf-8') as file:
    breed_class = json.load(file)

with open('sanitary.json', 'r', encoding='utf-8') as file:
    sanitary_class = json.load(file)

with open('layer.json', 'r', encoding='utf-8') as file:
    layer_class = json.load(file)

with open('category.json', 'r', encoding='utf-8') as file:
    category_class = json.load(file)

# Initialize session state to store marker location
if "marker_location" not in st.session_state:
    st.session_state.marker_location = [43.238949, 76.889709]  # Default location
    st.session_state.zoom = 11  # Default zoom
# lat = 43.238949
# lon = 76.889709

# Форма для ввода данных
age = st.number_input("Возраст (числовое значение)", min_value=0)  # Изменено на числовое значение
breed_id = st.selectbox("Порода насаждения", list(breed_class.keys()), format_func=lambda x: breed_class[x])
category_id = st.selectbox("Категория насаждения", list(category_class.keys()), format_func=lambda x: category_class[x])
layer_id = st.selectbox("Слой насаждения", list(layer_class.keys()), format_func=lambda x: layer_class[x])

# Функция предобработки данных
def preprocess_input(lat, lon, age, layer_id, category_id, breed_id):
    def int_to_multi(arg, class_dict):
        num_classes = len(class_dict)
        result = np.zeros(num_classes)
        i = 0
        for index, value in class_dict.items():
            if isinstance(arg, float):
                arg_cleaned = int(arg)  # Если arg — float, приводим его к int
            else:
                arg_cleaned = int(str(arg).split(";")[0])  # Если строка, разделяем по ";"
            if int(index) == arg_cleaned:
                result[i] = 1.
            i += 1
        return result

    # Числовые признаки
    numeric_data = np.array([lat, lon, age])  # age как числовое значение

    # One-hot кодирование для каждого категориального признака
    # layer_data = np.array([1 if layer_id == key else 0 for key in layer_class.keys()])
    # category_data = np.array([1 if category_id == key else 0 for key in category_class.keys()])
    # breed_data = np.array([1 if breed_id == key else 0 for key in breed_class.keys()])

    layer_data = int_to_multi(layer_id, layer_class)
    category_data = int_to_multi(category_id, category_class)
    breed_data = int_to_multi(breed_id, breed_class)

    # Добавление единицы для каждого вектора, если значение отсутствует
    # layer_data = np.append(layer_data, 1 if layer_id not in layer_class else 0)
    # category_data = np.append(category_data, 1 if category_id not in category_class else 0)
    # breed_data = np.append(breed_data, 1 if breed_id not in breed_class else 0)

    # Объединение всех признаков в одном массиве
    x_data = np.concatenate([numeric_data, layer_data, category_data, breed_data])

    return x_data.reshape(1, -1)

# Create the base map
m = folium.Map(location=st.session_state.marker_location, zoom_start=st.session_state.zoom)

# Add a marker at the current location in session state
marker = folium.Marker(
    location=st.session_state.marker_location,
    draggable=False
)
marker.add_to(m)

# Render the map and capture clicks
map = st_folium(m, width=620, height=580, key="folium_map")


# Update marker position immediately after each click
if map.get("last_clicked"):
    lat, lng = map["last_clicked"]["lat"], map["last_clicked"]["lng"]
    st.session_state.marker_location = [lat, lng]  # Update session state with new marker location
    st.session_state.zoom = map["zoom"]
    # Redraw the map immediately with the new marker location
    m = folium.Map(location=st.session_state.marker_location, zoom_start=st.session_state.zoom)
    folium.Marker(
        location=st.session_state.marker_location,
        draggable=False
    ).add_to(m)
    map = st_folium(m, width=620, height=580, key="folium_map")

# Display coordinates
st.write(f"Coordinates: {st.session_state.marker_location}")

# Кнопка для выполнения предсказания
if st.button("Предсказать"):
    input_data = preprocess_input(st.session_state.marker_location[0], st.session_state.marker_location[1], age, layer_id, category_id, breed_id)

    # Выполнение предсказания
    predictions = model.predict(input_data)[0]  # Получаем вероятности для каждого класса
    #sorted_classes = [sanitary_class.get(str(i), "Неизвестно") for i in predictions]

    # Вывод результата
    st.write("Предсказанные состояния и их вероятности:")
    for class_key, prob in zip(sanitary_class.keys(), predictions):
        state = sanitary_class[class_key]  # Получаем текстовое описание класса
        st.write(f"{state}: {prob * 100:.2f}%")

    # predicted_class = np.argmax(predictions, axis=1)[0]  # Получаем индекс класса
    # if predicted_class in sanitary_class:
    #     st.write(f"Прогнозируемое состояние: {sanitary_class[predicted_class]}")
    #     st.write(f"Предсказание: {predictions}")
    # else:
    #     st.write(f"Прогнозируемое состояние: Неизвестно. Возвращённый класс: {predicted_class}")