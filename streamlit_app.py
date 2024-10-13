import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("Iris Flower Classification")

# Load the data
@st.cache
def load_data():
    # Replace with your data path (can be from GitHub or local directory)
    df = pd.read_csv("Iris.csv")



    return df

df = load_data()

# Display dataset
st.write("### Dataset Preview")
st.write(df.head())

# Display basic statistics
st.write("### Dataset Statistics")
st.write(df.describe())

# Show missing values in the dataset
st.write("### Missing Values")
st.write(df.isnull().sum())

# Visualizations: Sepal Length vs Sepal Width by Species
fig_size = (10, 6)
fig, ax = plt.subplots(figsize=fig_size)
df[df.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='red', label='Setosa', ax=ax)
df[df.Species=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Versicolor', ax=ax)
df[df.Species=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Virginica', ax=ax)
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Sepal Length vs Sepal Width")
st.pyplot(fig)

# Visualize Petal Length vs Petal Width by Species
fig, ax = plt.subplots(figsize=fig_size)
df[df.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Setosa', ax=ax)
df[df.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Versicolor', ax=ax)
df[df.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Virginica', ax=ax)
ax.set_xlabel("Petal Length")
ax.set_ylabel("Petal Width")
ax.set_title("Petal Length vs Petal Width")
st.pyplot(fig)

# Swarm Plots to visualize feature distribution by species
st.write("### Swarm Plots of Features by Species")
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Petal Length
sns.swarmplot(x='Species', y='PetalLengthCm', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Swarm Plot of Petal Length by Species')

# Petal Width
sns.swarmplot(x='Species', y='PetalWidthCm', data=df, ax=axs[0, 1])
axs[0, 1].set_title('Swarm Plot of Petal Width by Species')

# Sepal Length
sns.swarmplot(x='Species', y='SepalLengthCm', data=df, ax=axs[1, 0])
axs[1, 0].set_title('Swarm Plot of Sepal Length by Species')

# Sepal Width
sns.swarmplot(x='Species', y='SepalWidthCm', data=df, ax=axs[1, 1])
axs[1, 1].set_title('Swarm Plot of Sepal Width by Species')

plt.tight_layout()
st.pyplot(fig)

# Prepare data for training
train, test = train_test_split(df, test_size=0.3, random_state=42)
train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
train_y = train.Species
test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
test_y = test.Species

# Encode target labels
label_encoder = LabelEncoder()
train_y_encoded = label_encoder.fit_transform(train_y)
test_y_encoded = label_encoder.transform(test_y)

# Train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(train_X, train_y_encoded)

# Model prediction
prediction = model.predict(test_X)
accuracy = metrics.accuracy_score(prediction, test_y_encoded)

# Display model accuracy
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Allow users to make predictions with the trained model
st.write("### Make Predictions on New Data")

# User inputs for prediction
sepal_length = st.number_input("Enter Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Enter Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Enter Petal Length (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_width = st.number_input("Enter Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.0)

# Create a DataFrame for the input values
input_data = pd.DataFrame({
    'SepalLengthCm': [sepal_length],
    'SepalWidthCm': [sepal_width],
    'PetalLengthCm': [petal_length],
    'PetalWidthCm': [petal_width]
})

# Make a prediction for the user input
predicted_class = model.predict(input_data)
predicted_species = label_encoder.inverse_transform(predicted_class)

# Display prediction result
st.write(f"The predicted species for the given input is: {predicted_species[0]}")
