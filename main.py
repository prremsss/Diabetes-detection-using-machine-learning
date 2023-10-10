import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

image = Image.open('D:/Projet python/logo.png')
st.image(image, use_column_width=True)
st.markdown("<h1 style='text-align: center; color:#1278a9 ;'>Detection de diabète</h1>", unsafe_allow_html=True)
image1 = Image.open('D:/Projet python/3701981.jpg')
st.image(image1, use_column_width=True)
# load data
data = pd.read_csv('D:/Projet python/diabetes.csv')
data1 = pd.read_csv('D:/Projet python/diabetes.csv')

# remplacement des 0 par nan dans 'Glucose','BloodPressure','SkinThickness','Insulin','BMI'
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
# remplacement de nan par des valeurs convenables
data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].mean(), inplace=True)
data['Insulin'].fillna(data['Insulin'].mean(), inplace=True)
data['BMI'].fillna(data['BMI'].mean(), inplace=True)
choice = st.sidebar.selectbox("", ("Entraînement", "Prediction"))

# splitting the data

X = data.iloc[:, 0:8].values
Y = data.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
test_scores = []
train_scores = []

for i in range(1, 10):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, Y_train)

    train_scores.append(knn.score(X_train, Y_train))
    test_scores.append(knn.score(X_test, Y_test))
k = np.argmax(test_scores) + 1

if choice == "Entraînement":
    st.markdown("<h1 style='text-align: center; color:#f59116 ;'>Entraînement</h1>", unsafe_allow_html=True)
    # affichage des données
    st.subheader('Données')
    st.dataframe(data.head())
    st.subheader('Informations sur les données')
    st.write(data.describe())
    # Visualisation des données
    st.subheader('Visualisation des données')

    data.hist(figsize=(20, 20))
    st.pyplot()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.pairplot(data, hue='Outcome')
    st.pyplot()


    st.subheader("Scores d'Entraînement et de test")

    plt.figure()
    sns.lineplot(range(1, 10), train_scores, marker='*', label='Train Score')
    sns.lineplot(range(1, 10), test_scores, marker='o', label='Test Score')
    st.pyplot()


else:
    st.markdown("<h1 style='text-align: center; color:#f59116 ;'>Prediction</h1>", unsafe_allow_html=True)


    def user_input():
        age = st.sidebar.slider('Age', 21, 81, 33)
        pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 0)
        glucose = st.sidebar.slider('Glucose', 44, 199, 199)
        bloodPressure = st.sidebar.slider('BloodPressure', 24, 122, 81)
        skinThickness = st.sidebar.slider('SkinThickness', 7, 99, 66)
        insulin = st.sidebar.slider('Insulin', 14, 846, 846)
        bmi = st.sidebar.slider('Bmi', 18.2, 67.1, 43.14)
        diabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.0780, 2.4200, 2.24)
        _data = {'age': age,
                 'pregnancies': pregnancies,
                 'glucose': glucose,
                 'bloodPressure': bloodPressure,
                 'skinThickness': skinThickness,
                 'insulin': insulin,
                 'bmi': bmi,
                 'diabetesPedigreeFunction': diabetesPedigreeFunction
                 }
        features = pd.DataFrame(_data, index=[0])
        return features

    user_input = user_input()
    if st.sidebar.button("Predire"):

        st.subheader("Données d'utilisateur")
        st.write(user_input)
        prediction = knn.predict(user_input)
        st.subheader("Résultats:")


        if prediction==0:
            st.write("Heureusement vous n'êtes pas diabétique")
        else :
            st.write("Malheureusement il y'a un risque très élevé que vous soyez diabétique")

