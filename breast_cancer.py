import pickle
import pandas as pd
import streamlit as st
import numpy as np
from joblib import dump, load

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
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction']) #two pages

if app_mode=='Home':
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
    st.markdown("- **fractal_dimension_worst**: Worst Fractal Dimension")
    st.markdown("- **concave_points_worst**: Worst Concave Points")
    st.markdown("- **compactness_worst**: Worse Compactness")
    st.markdown("- **concavity_mean**: Mean of Concavity")
    st.markdown("- **compactness_mean**: Mean of Compactness")
    st.markdown("- **radius_se**: SE of Radius")
    st.markdown("- **compactness_se**: SE of compactness")
    st.markdown("- **symmetry_worst**: Worst Symmetry")

    # Pandasla veri setini okuyalım
    df = pd.read_csv("breast-cancer.csv")
    # Tablo Ekleme
    st.table( df.sample(5, random_state=42))
elif app_mode == 'Prediction':
    st.sidebar.subheader("*:blue[Let's try classifying these tumors using machine learning!]*")

    compactness_mean = st.sidebar.number_input("Mean of Compactness", format="%.4f")
    concavity_mean = st.sidebar.number_input("Mean of Concavity", format="%.4f")
    radius_se = st.sidebar.number_input("SE of Radius", format="%.4f")
    compactness_se = st.sidebar.number_input("SE of Compactness", format="%.4f")
    compactness_worst = st.sidebar.number_input("Worst Compactness", format="%.4f")
    concave_points_worst = st.sidebar.number_input("Worst Concave Points", format="%.4f")
    symmetry_worst = st.sidebar.number_input("Worst Symmetry", format="%.4f")
    fractal_dimension_worst = st.sidebar.number_input("Worst Fractal Dimension", format="%.4f")
    # st.sidebar.button("Submit")
    # Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması


    log_reg = load('logreg_model.pkl')

    input_df = pd.DataFrame({
        'compactness_mean':[compactness_mean],
        'concavity_mean':[concavity_mean],
        'radius_se':[radius_se],
        'compactness_se':[compactness_se],
        'compactness_worst':[compactness_worst],
        'concave points_worst':[concave_points_worst],
        'symmetry_worst':[symmetry_worst],
        'fractal_dimension_worst':[fractal_dimension_worst]

    }, index=[0])

    pred = log_reg.predict(input_df.values)
    pred_probability = np.round(log_reg.predict_proba(input_df.values), 2)

    #---------------------------------------------------------------------------------------------------------------------

    st.header("**:red[Results:]**")

    # Sonuç Ekranı
    if st.sidebar.button("Submit"):

        # Info mesajı oluşturma
        st.info("*You can find the result below.*")

        # Sonuçları Görüntülemek için DataFrame
        results_df = pd.DataFrame({
            'compactness_mean':[compactness_mean],
            'concavity_mean':[concavity_mean],
            'radius_se':[radius_se],
            'compactness_se':[compactness_se],
            'compactness_worst':[compactness_worst],
            'concave points_worst':[concave_points_worst],
            'symmetry_worst':[symmetry_worst],
            'fractal_dimension_worst':[fractal_dimension_worst],
            'Prediction': [pred]

        }, index=[0])

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
