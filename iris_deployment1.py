#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42
)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Compute accuracy
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Streamlit UI
st.title("Decision Tree Classifier-iris Data")
st.write(f"Model Accuracy: {accuracy:.2f}")

st.sidebar.header("Input feature (data)")
sepal_length = st.sidebar.slider("Sepal Length", float(df.iloc[:, 0].min()), float(df.iloc[:, 0].max()), float(df.iloc[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(df.iloc[:, 1].min()), float(df.iloc[:, 1].max()), float(df.iloc[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(df.iloc[:, 2].min()), float(df.iloc[:, 2].max()), float(df.iloc[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(df.iloc[:, 3].min()), float(df.iloc[:, 3].max()), float(df.iloc[:, 3].mean()))

# Prediction
data_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(data_input)
predicted_class = data.target_names[prediction[0]]

st.subheader("Prediction")
st.write(f"Predicted Class: **{predicted_class}**")

import io
if st.button("Show Decision Tree"):
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Decision Tree Visualization", use_column_width=True)


