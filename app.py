import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Predicción de Adopción de Mascotas",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Cargar Modelo y Preprocesadores ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('best_gbc_model.joblib')
        onehot_encoder = joblib.load('onehot_encoder.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return model, onehot_encoder, label_encoder
    except FileNotFoundError:
        st.error("¡Error! Archivos del modelo o preprocesadores no encontrados. Asegúrate de que 'best_gbc_model.joblib', 'onehot_encoder.joblib' y 'label_encoder.joblib' estén en el directorio correcto.")
        st.stop()

model, onehot_encoder, label_encoder = load_resources()

# --- Título y Descripción de la Aplicación ---
st.title("🐾 Predicción de Adopción de Mascotas")
st.markdown("Esta aplicación te ayuda a predecir si una mascota tiene alta o baja probabilidad de ser adoptada, basándose en sus características. ¡Ayudemos a encontrarles un hogar!", unsafe_allow_html=False)
st.markdown("--- ")

# --- Diseño de la Interfaz (UX/UI) ---

# Sidebar para información adicional o branding
st.sidebar.image("https://i.ibb.co/3W6qWj6/paw-print-1.png", width='stretch', caption="¡Adopta, no compres!") # Placeholder image
st.sidebar.header("Sobre la Predicción")
st.sidebar.info(
    "Los resultados se basan en un modelo de Machine Learning entrenado con datos históricos de adopciones. "
    "Un resultado de 'Alta Probabilidad' sugiere que la mascota cumple con características comúnmente asociadas a una adopción exitosa, mientras que 'Baja Probabilidad' indica lo contrario. "
    "¡Cada mascota es única y merece una oportunidad!"
)

# Colores temáticos para adopción de mascotas (ej. tonos tierra, verde, azul suave)
# Puedes usar CSS personalizado si necesitas más control
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f3e9; /* Light beige/cream */
    }
    .css-1d391kg, .css-1dp5r7x {
        background-color: #e0e7d6; /* Soft green for sidebar/elements */
    }
    .st-bu {
        background-color: #8bb4a2; /* Muted teal for buttons */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Campos de Entrada del Usuario ---
st.header("Ingresa las Características de la Mascota:")

# Input para PetType
pet_type = st.selectbox(
    "Tipo de Mascota",
    ('Dog', 'Cat', 'Bird', 'Rabbit'),
    help="Selecciona el tipo de mascota."
)

# Input para AgeMonths
age_months = st.slider(
    "Edad en Meses",
    min_value=1, max_value=200, value=12,
    help="Ingresa la edad de la mascota en meses."
)

# Input para Size
size = st.selectbox(
    "Tamaño",
    ('Small', 'Medium', 'Large'),
    help="Selecciona el tamaño de la mascota."
)

# Input para Vaccinated
vaccinated_map = {'Sí': 1, 'No': 0}
vaccinated_input = st.radio(
    "¿Está vacunada?",
    ('Sí', 'No'),
    help="Indica si la mascota ha sido vacunada."
)
vaccinated = vaccinated_map[vaccinated_input]

# Input para HealthCondition
health_condition_map = {'Sano': 0, 'Problema de salud': 1}
health_condition_input = st.radio(
    "Condición de Salud",
    ('Sano', 'Problema de salud'),
    help="Indica la condición de salud de la mascota."
)
health_condition = health_condition_map[health_condition_input]

# --- Función de Preprocesamiento ---
def preprocess_input(pet_type, age_months, size, vaccinated, health_condition, onehot_enc, label_enc):
    # Crear DataFrame inicial con las mismas columnas que el entrenamiento
    # Aunque solo PetType y Size serán One-Hot encoded, se necesitan todas las columnas para el `transform`
    data = {
        'PetType': [pet_type],
        'AgeMonths': [age_months],
        'Size': [size],
        'Vaccinated': [vaccinated],
        'HealthCondition': [health_condition]
    }
    df_input = pd.DataFrame(data)

    # Convertir 'PetType' y 'Size' a tipo 'category'
    df_input['PetType'] = df_input['PetType'].astype('category')
    df_input['Size'] = df_input['Size'].astype('category')

    # Aplicar One-Hot Encoding con el encoder cargado
    onehot_cols = ['PetType', 'Size']
    onehot_encoded = onehot_enc.transform(df_input[onehot_cols])
    onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_enc.get_feature_names_out(onehot_cols))

    # Concatenar y eliminar columnas originales One-Hot encoded
    df_processed = pd.concat([df_input.drop(columns=onehot_cols), onehot_df], axis=1)

    # Reordenar las columnas para que coincidan con X_train del modelo
    # Se asume que X_train tiene las columnas en el orden correcto.
    # Si no, se debería cargar X_train.columns para asegurar el orden.
    # Para este ejemplo, usaremos un orden predefinido basado en el entrenamiento previo.
    expected_columns = [
        'AgeMonths', 'Vaccinated', 'HealthCondition', 'PetType_Bird',
        'PetType_Cat', 'PetType_Dog', 'PetType_Rabbit', 'Size_Large',
        'Size_Medium', 'Size_Small'
    ]

    # Asegurarse de que todas las columnas esperadas estén presentes y en el orden correcto
    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0  # Añadir columnas que podrían faltar (ej. si no hay un PetType_Bird en el input pero sí en el entrenamiento)

    df_processed = df_processed[expected_columns]

    return df_processed

# --- Botón de Predicción ---
if st.button("Predecir Probabilidad de Adopción"):
    processed_data = preprocess_input(pet_type, age_months, size, vaccinated, health_condition, onehot_encoder, label_encoder)
    prediction = model.predict(processed_data)[0]

    st.markdown("## Resultado de la Predicción:")
    if prediction == 1:
        st.success(
            f"### ¡Excelente noticia! Esta mascota tiene una **ALTA PROBABILIDAD** de ser adoptada. 🎉"
        )
        st.balloons()
        st.markdown(
            "<p style='font-size:18px;'>Sus características sugieren que es muy deseable para adopción. "
            "¡Esperemos que encuentre un hogar pronto!</p>",
            unsafe_allow_html=True
        )
    else:
        st.info(
            f"### Esta mascota tiene una **BAJA PROBABILIDAD** de ser adoptada. 😔"
        )
        st.markdown(
            "<p style='font-size:18px;'>Aunque las características actuales no son las más favorables para la adopción "
            "según el modelo, cada mascota tiene su encanto. Considera acciones para "
            "aumentar sus posibilidades, como campañas de difusión, capacitación o "
            "atención médica adicional.</p>",
            unsafe_allow_html=True
        )
    st.markdown("--- ")
    st.markdown("### ¡Ayúdanos a encontrarle un hogar a cada mascota!")

# --- Footer o Información Adicional ---
st.markdown(
    """
    <br><br>
    <p style='text-align: center; color: gray;'>
        Desarrollado con Streamlit para el despliegue de modelos de Machine Learning. <br>
        &copy; 2024 Proyecto Adopción de Mascotas.
    </p>
    """,
    unsafe_allow_html=True
)
