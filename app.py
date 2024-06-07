from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the preprocessed data
df = pd.read_csv('mobil.csv')

# Drop non-numeric columns
if 'Name' in df.columns:
    df = df.drop('Name', axis=1)
if 'Location' in df.columns:
    df = df.drop('Location', axis=1)
if 'New_Price' in df.columns:
    df = df.drop('New_Price', axis=1)

# Handle categorical columns using one-hot encoding
categorical_columns = ['Transmission', 'Fuel_Type', 'Owner_Type']
for col in categorical_columns:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

# Convert 'Mileage' column to numeric
df['Mileage'] = df['Mileage'].str.split(' ').str[0].astype(float)

# Convert 'Engine' column to numeric
df['Engine'] = df['Engine'].str.split(' ').str[0].astype(float)

# Convert 'Power' column to numeric
df['Power'] = df['Power'].str.split(' ').str[0]
df['Power'] = pd.to_numeric(df['Power'], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Split the dataset into features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Train a Random Forest Regression model
random_forest = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=5)
random_forest.fit(X_train, y_train)

# Train a Ridge Regression model
ridge = Ridge()
ridge.fit(X_train, y_train)

# Train a Lasso Regression model
lasso = Lasso()
lasso.fit(X_train, y_train)

# Train an Elastic-Net Regression model
elastic_net = ElasticNet()
elastic_net.fit(X_train, y_train)

# Define a function for evaluation metrics
def evaluation(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_true, y_pred)
    return mae, mse, rmse, r_squared

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for predicting car prices
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    year = int(request.form['year'])
    kilometers_driven = int(request.form['kilometers_driven'])
    # Add more input variables as needed

    # Prepare the input data for prediction
    input_data = [[year, kilometers_driven]]  # Add more features as needed

    # Make predictions using the trained models
    lin_reg_prediction = lin_reg.predict(X_test)  # Use X_test instead of input_data
    random_forest_prediction = random_forest.predict(X_test)  # Use X_test instead of input_data
    ridge_prediction = ridge.predict(X_test)  # Use X_test instead of input_data
    lasso_prediction = lasso.predict(X_test)  # Use X_test instead of input_data
    elastic_net_prediction = elastic_net.predict(X_test)  # Use X_test instead of input_data

    return render_template('result.html', 
                            lin_reg_prediction=lin_reg_prediction,
                            random_forest_prediction=random_forest_prediction,
                            ridge_prediction=ridge_prediction,
                            lasso_prediction=lasso_prediction,
                            elastic_net_prediction=elastic_net_prediction)

if __name__ == '__main__':
    app.run(debug=True)
