import pickle
import pandas as pd
import streamlit as st
import numpy as np
from joblib import load

# Sayfa Ayarları


st.set_page_config(
    page_title="Breast Tumor Classification",
    page_icon="",
    layout="centered",
    menu_items={
        "Get help": "mailto:nygulzehra@gmail.com",
        "About": "For More Information\n" + "https://github.com/nygulzehra"
    }
)

st.image("bcancer.jpeg")
# Başlık Ekleme
st.title("**:red[Breast Tumor Classification Project]**")

# Markdown Oluşturma
st.markdown("Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.")

st.markdown("The key challenges against it’s detection is how to classify tumors into malignant (cancerous) or benign(non cancerous). ")
st.markdown("Thanks to last machine learning technologies, we can predict tumors as benign or malignant beforehand.")

st.markdown("We used Breast Cancer Wisconsin (Diagnostic) Dataset.")

# Resim Ekleme
st.image("bc-image-cover.png")



# Header Ekleme
st.header("**:red[Breast Cancer Dateset Dictionary]**:")
st.markdown("- **area_worst**: Worst Area")
st.markdown("- **concave_points_worst**: Worst Concave Points")
st.markdown("- **texture_worst**: Worst Texture")
st.markdown("- **texture_mean**: Mean of Texture")
st.markdown("- **concavity_worst**: Worst Concavity")
st.markdown("- **symmetry_se**: SE of Symmetry")
st.markdown("- **radius_worst**: Worst Radius")
st.markdown("- **compactness_worst**: Worse Compactness")
st.markdown("- **fractal_dimension_worst**: Worst Fractal Dimension")
st.markdown("- **fractal_dimension_se**: SE of Fractal Dimension")
st.markdown("- **concavity_mean**: Mean of Concavity")
st.markdown("- **area_se**: Area of SE")
st.markdown("- **smoothness_se**: SE of Smoothness")

# st.markdown("- **area_mean**: Mean Area of Lobes")
# st.markdown("- **smoothness_mean**: Mean of Smoothness Levels")
# st.markdown("- **compactness_mean**: Mean of Compactness")
# st.markdown("- **concave points_mean**: Mean of Cocave Points")
# st.markdown("- **symmetry_mean**: Mean of Symmetry")
# st.markdown("- **fractal_dimension_mean**: Mean of Fractal Dimension")
# st.markdown("- **radius_se**: SE of Radius")
# st.markdown("- **texture_se**: SE of Texture")
# st.markdown("- **perimeter_se**: Perimeter of SE")
# st.markdown("- **compactness_se**: SE of compactness")
# st.markdown("- **concavity_se**: SEE of concavity")
# st.markdown("- **concave points_se**: SE of concave points")
# st.markdown("- **perimeter_worst**: Worst Permimeter")
# st.markdown("- **smoothness_worst**: Worst Smoothness")
# st.markdown("- **symmetry_worst**: Worst Symmetry")

# Pandasla veri setini okuyalım
df = pd.read_csv("breast-cancer.csv")
# Tablo Ekleme
st.table( df.sample(5, random_state=42))

st.sidebar.subheader("*:blue[Let's try classifying these tumors using machine learning!]*")

area_worst = st.sidebar.number_input("Worst Area", format="%.4f")
concave_points_worst = st.sidebar.number_input("Worst Concave Points", format="%.4f")
texture_worst = st.sidebar.number_input(label="Worst Texture", format="%.4f")
texture_mean = st.sidebar.number_input(label="Mean of Texture", format="%.4f")
concavity_worst = st.sidebar.number_input(label="Worst Concavity ", format="%.4f")
symmetry_se = st.sidebar.number_input(label="SE of Symmetry", format="%.4f")
radius_worst = st.sidebar.number_input(label="Worst Radius", format="%.4f")
compactness_worst = st.sidebar.number_input(label="Worse Compactness ", format="%.4f")
fractal_dimension_worst = st.sidebar.number_input(label="Worst Fractal Dimension ", format="%.4f")
fractal_dimension_se = st.sidebar.number_input(label="SE of Fractal Dimension", format="%.4f")
concavity_mean = st.sidebar.number_input(label="Mean of Concavity", format="%.4f")
area_se = st.sidebar.number_input(label="Area of SE ", format="%.4f")
smoothness_se = st.sidebar.number_input(label="SE of Smoothness", format="%.4f")
# st.sidebar.button("Submit")

# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması


dt2_model = load('dt2_model.pkl')

input_df = pd.DataFrame({
    'area_worst' : [area_worst],
    'concave_points_worst' : [concave_points_worst],
    'texture_worst' : [texture_worst],
    'texture_mean' : [texture_mean],
    'concavity_worst' : [concavity_worst],
    'symmetry_se' : [symmetry_se],
    'radius_worst' : [radius_worst],
    'compactness_worst' : [compactness_worst],
    'fractal_dimension_worst' : [fractal_dimension_worst],
    'fractal_dimension_se' : [fractal_dimension_se],
    'concavity_mean' : [concavity_mean],
    'area_se' : [area_se],
    'smoothness_se' : [smoothness_se]

})

pred = dt2_model.predict(input_df.values)
pred_probability = np.round(dt2_model.predict_proba(input_df.values), 2)

#---------------------------------------------------------------------------------------------------------------------

st.header("**:red[Results:]**")

# Sonuç Ekranı
if st.sidebar.button("Submit"):

    # Info mesajı oluşturma
    st.info("*You can find the result below.*")

    # Sonuçları Görüntülemek için DataFrame
    results_df = pd.DataFrame({
        'area_worst' : [area_worst],
        'concave_points_worst' : [concave_points_worst],
        'texture_worst' : [texture_worst],
        'texture_mean' : [texture_mean],
        'concavity_worst' : [concavity_worst],
        'symmetry_se' : [symmetry_se],
        'radius_worst' : [radius_worst],
        'compactness_worst' : [compactness_worst],
        'fractal_dimension_worst' : [fractal_dimension_worst],
        'fractal_dimension_se' : [fractal_dimension_se],
        'concavity_mean' : [concavity_mean],
        'area_se' : [area_se],
        'smoothness_se' : [smoothness_se],
        'Prediction': [pred]

    })

    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("0","Benign"))
    results_df["Prediction"] = results_df["Prediction"].apply(lambda x: str(x).replace("1","Malignant"))

    st.table(results_df)

    if pred == 0:
        st.markdown('**Yay!!!!**')
        st.markdown('**It is Benign !!!**')
        st. balloons ()
    else:
        st.markdown('**It is look like Malignant !!!**')
else:
    st.markdown("Please click the *Submit Button* to see the result!")
