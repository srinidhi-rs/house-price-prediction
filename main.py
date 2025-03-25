import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
file_path = "USA_Housing.csv"  # Ensure the correct file path
HouseDF = pd.read_csv(file_path)

# Select relevant features
X = HouseDF[['Avg. Area Income', 'Area Population', 'Avg. Area Number of Rooms']]
y = HouseDF['Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Function to predict house price based on user input
def predict_house_price():
    try:
        income = float(input("Enter the average area income: "))
        population = float(input("Enter the area population: "))
        rooms = float(input("Enter the average number of rooms: "))

        input_data = np.array([[income, population, rooms]])
        predicted_price = model.predict(input_data)[0]
        print(f"Predicted House Price: ${predicted_price:,.2f}")
    except ValueError:
        print("Invalid input. Please enter numeric values.")


# Run the prediction function
if __name__ == "__main__":
    predict_house_price()
