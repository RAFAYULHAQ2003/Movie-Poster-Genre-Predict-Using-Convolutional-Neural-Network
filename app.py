import streamlit as st
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load and preprocess image
def load_and_preprocess_image(file_path, target_size=(340, 340)):
    img = image.load_img(file_path, target_size=target_size)
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

st.title('Image Genre Prediction')

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Missing Values
    data.dropna(inplace=True)

    # Load image data
    width = 340
    height = 340
    X = []
    for i in range(data.shape[0]):
        path = 'D:/PythonProj/RafayAI/RAFAY/archive (4)/Multi_Label_dataset/Images/'+data['Id'][i]+'.jpg'
        img = load_and_preprocess_image(path, target_size=(width, height))
        X.append(img)

    X = np.array(X)
    y = data.drop(['Id', 'Genre'], axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define model
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=X_train[0].shape),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.3),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.3),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(2, 2),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(25, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # File upload for image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = load_and_preprocess_image(uploaded_image)

        # Show the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        prediction = model.predict(img)
        classes = data.columns[2:]
        top3 = np.argsort(prediction[0])[:-4:-1]

        st.write("Top 3 predicted genres:")
        for i in range(3):
            st.write(classes[top3[i]])
