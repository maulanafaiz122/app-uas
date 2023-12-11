import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load scaler and model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('best_rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(input_data):
    # Encode categorical features using LabelEncoder
    encoder = LabelEncoder()
    for column in input_data.select_dtypes(include='object').columns:
        input_data[column] = encoder.fit_transform(input_data[column])

    # Ensure numerical features match those used during model training
    numerical_features = [
        'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
        'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 
        'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 
        'population', 'habitat'
    ]  # List all numerical features used during training

    # Select only the required columns excluding 'class'
    input_data = input_data[numerical_features]

    # Normalize numerical features using the loaded scaler
    input_data = scaler.transform(input_data)

    return input_data

# Streamlit app
# Form input for mushroom features
st.write('Mohon masukkan fitur-fitur jamur:')
form = st.form(key='mushrooms_update.csv')
cap_shape = form.text_input('Bentuk Tutup (Enter a value between 0 and 5)', '0')
cap_surface = form.text_input('Permukaan Tutup (Enter a value between 0 and 3)', '0')
cap_color = form.text_input('Warna Tutup (Enter a value between 0 and 9)', '0')
bruises = form.text_input('Lecet (Enter 0 for No, 1 for Yes)', '0')
odor = form.text_input('Bau (Enter a value between 0 and 8)', '0')
gill_attachment = form.text_input('Lampiran Gill (Enter a value between 0 and 3)', '0')
gill_spacing = form.text_input('Jarak Gill (Enter a value between 0 and 2)', '0')
gill_size = form.text_input('Ukuran Gill (Enter 0 for Small, 1 for Large)', '0')
gill_color = form.text_input('Warna Gill (Enter a value between 0 and 11)', '0')
stalk_shape = form.text_input('Bentuk Stalk (Enter 0 for Enlarging, 1 for Tapering)', '0')
stalk_root = form.text_input('Akar Stalk (Enter a value between 0 and 6)', '0')
stalk_surface_above_ring = form.text_input('Permukaan Stalk di Atas Cincin (Enter a value between 0 and 3)', '0')
stalk_surface_below_ring = form.text_input('Permukaan Stalk di Bawah Cincin (Enter a value between 0 and 3)', '0')
stalk_color_above_ring = form.text_input('Warna Stalk di Atas Cincin (Enter a value between 0 and 8)', '0')
stalk_color_below_ring = form.text_input('Warna Stalk di Bawah Cincin (Enter a value between 0 and 7)', '0')
veil_type = form.text_input('Tipe Veil (Enter 0 for Partial, 1 for Universal)', '0')
veil_color = form.text_input('Warna Veil (Enter a value between 0 and 3)', '0')
ring_number = form.text_input('Jumlah Cincin (Enter a value between 0 and 3)', '0')
ring_type = form.text_input('Tipe Cincin (Enter a value between 0 and 7)', '0')
spore_print_color = form.text_input('Warna Spore Print (Enter a value between 0 and 8)', '0')
population = form.text_input('Populasi (Enter a value between 0 and 5)', '0')
habitat = form.text_input('Habitat (Enter a value between 0 and 6)', '0')

# Add this line to define the submit_button variable
submit_button = form.form_submit_button(label='Deteksi Jenis Jamur')

# If there's a form submission
if submit_button:
    # Create a DataFrame with user input
    input_data = pd.DataFrame({
        'class': [0],  # Add a placeholder target column
        'cap-shape': [cap_shape],
        'cap-surface': [cap_surface],
        'cap-color': [cap_color],
        'bruises': [bruises],
        'odor': [odor],
        'gill-attachment': [gill_attachment],
        'gill-spacing': [gill_spacing],
        'gill-size': [gill_size],
        'gill-color': [gill_color],
        'stalk-shape': [stalk_shape],
        'stalk-root': [stalk_root],
        'stalk-surface-above-ring': [stalk_surface_above_ring],
        'stalk-surface-below-ring': [stalk_surface_below_ring],
        'stalk-color-above-ring': [stalk_color_above_ring],
        'stalk-color-below-ring': [stalk_color_below_ring],
        'veil-type': [veil_type],
        'veil-color': [veil_color],
        'ring-number': [ring_number],
        'ring-type': [ring_type],
        'spore-print-color': [spore_print_color],
        'population': [population],
        'habitat': [habitat],
    })

    # Process input data
    preprocessed_data = preprocess_input(input_data)

    # Make predictions using the trained model
    prediction = rf_model.predict(preprocessed_data)

    # Display the result
    st.subheader('Prediksi:')
    if prediction[0] == 1:
        st.write('Jamur diprediksi **beracun**.')
    else:
        st.write('Jamur diprediksi **aman dikonsumsi**.')
