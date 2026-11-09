
# 🏠 House Price Prediction Using Multiple Linear Regression

## 📌 Project Overview

House Price Prediction is a Machine Learning project that predicts the estimated selling price of a house based on different property-related features.

This project uses **Multiple Linear Regression**, a supervised Machine Learning algorithm, to predict house prices using property details such as bedrooms, bathrooms, living area, lot area, floors, house condition, location, and other features.

The model is developed using Python and Scikit-learn and deployed as a web application using Flask, HTML, and CSS.

## 🎯 Project Objectives

- Predict house prices based on property details.
- Load and process a real-world housing dataset.
- Perform categorical data encoding.
- Identify independent and dependent variables.
- Split the dataset into training and testing data.
- Train a Multiple Linear Regression model.
- Evaluate the model using accuracy and loss calculations.
- Save the trained model using Pickle.
- Integrate the trained model with a Flask web application.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization library |
| Scikit-learn | Machine Learning |
| Linear Regression | Prediction algorithm |
| Pickle | Saving and loading the trained model |
| Flask | Web application development |
| HTML | Frontend structure |
| CSS | Frontend styling |

## 📂 Project Structure

```text
House-Price-Prediction/
│
├── data.csv
│
├── main.py
│
├── app.py
│
├── Model.pkl
│
├── templates/
│   └── index.html
│
└── README.md
```

### File Description

| File | Description |
|---|---|
| `data.csv` | Dataset containing house details and selling prices |
| `main.py` | Performs preprocessing, model training, testing, evaluation, and model saving |
| `app.py` | Flask application for predicting house prices |
| `Model.pkl` | Saved trained Linear Regression model |
| `index.html` | Frontend interface for entering property details |
| `README.md` | Project documentation |

## 📊 Dataset Description

The project uses a housing dataset named `data.csv`.

**Dataset Information:**
- Total Records: 4,600
- Total Columns: 18
- Target Variable: `price`

### Dataset Features

| Column Name | Description |
|---|---|
| date | Date of house sale |
| price | Selling price of the house |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| sqft_living | Living area in square feet |
| sqft_lot | Lot area in square feet |
| floors | Number of floors |
| waterfront | Waterfront availability |
| view | View rating |
| condition | Condition of the house |
| sqft_above | Area above ground |
| sqft_basement | Basement area |
| yr_built | Year of construction |
| yr_renovated | Year of renovation |
| street | Street address |
| city | City name |
| statezip | State ZIP code |
| country | Country name |

## 📌 Independent and Dependent Variables

### 1. Independent Variables (X)

Independent variables are the input features used to predict the house price.

The project uses the following 15 independent variables:

1. bedrooms
2. bathrooms
3. sqft_living
4. sqft_lot
5. floors
6. waterfront
7. view
8. condition
9. sqft_above
10. sqft_basement
11. yr_built
12. yr_renovated
13. city
14. statezip
15. country

### 2. Dependent Variable (Y)

The dependent variable is the target value predicted by the Machine Learning model.

- `price` – Selling price of the house.

### Feature Selection in Python

```python
self.X = df.iloc[:, 1:]
self.y = df.iloc[:, 0]
```

- `X` represents the independent variables.
- `y` represents the dependent variable (house price).

## ⚙️ Project Implementation

### 1. Importing Required Libraries

The following libraries are imported in `main.py`:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
```

### 2. Loading the Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("data.csv")
```

This loads the house price dataset into a Pandas DataFrame.

### 3. Removing Unnecessary Columns

The `date` and `street` columns are removed because they are not used as input features in the current model.

```python
df = df.drop(['date', 'street'], axis=1)
```

### 4. Categorical Data Encoding

Machine Learning algorithms require numerical input values. Therefore, categorical columns are converted into numerical representations.

**City Encoding**

First, unique city names are extracted.

```python
l = df['city'].unique()
d = {}

for i in range(len(l)):
    d[l[i]] = i

df['city'] = df['city'].map(d)
```

Each unique city is assigned a numerical value based on its position in the unique city list.

**State ZIP Encoding**

Unique state ZIP values are extracted and converted into numerical values.

```python
m = df['statezip'].unique()
n = {}

for j in range(len(m)):
    n[m[j]] = j

df['statezip'] = df['statezip'].map(n)
```

**Country Encoding**

The country column is converted into numerical form.

```python
df['country'] = df['country'].map({'USA': 0}).astype(int)
```

The encoding is:

```text
USA → 0
```

After encoding, all the features used by the model are numerical.

## 📌 5. Selecting Independent and Dependent Variables

The dataset is divided into input features and the target variable.

```python
self.X = df.iloc[:, 1:]
self.y = df.iloc[:, 0]
```

- Independent variables contain the property details.
- Dependent variable contains the house selling price.

## 📚 6. Splitting Training and Testing Data

The dataset is divided into training and testing datasets using Scikit-learn's `train_test_split()`.

```python
self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
    self.X,
    self.y,
    test_size=0.2,
    random_state=42
)
```

**Dataset splitting:**

- 80% – Training Data
- 20% – Testing Data
- Random State – 42

The training dataset is used to train the model, while the testing dataset is used to evaluate its performance on unseen data.

## 🤖 7. Machine Learning Algorithm

### Multiple Linear Regression

Multiple Linear Regression is a supervised Machine Learning algorithm used to predict continuous numerical values based on multiple independent variables.

In this project, the algorithm learns the relationship between different property features and their selling prices.

**General Mathematical Equation:**

```text
Y = b0 + b1X1 + b2X2 + b3X3 + ... + bnXn
```

Where:

- Y = Predicted house price
- b0 = Intercept
- b1, b2, ..., bn = Regression coefficients
- X1, X2, ..., Xn = Independent variables

## 🏋️ 8. Model Training

The Linear Regression model is created and trained using the training dataset.

```python
self.reg = LinearRegression()

self.reg.fit(self.X_train, self.y_train)
```

After training, predictions are generated for the training dataset.

```python
train_predictions = self.reg.predict(self.X_train)
```

The model learns the relationship between the property features and their corresponding house prices.

## 🔍 9. Model Testing

The trained model is tested using the testing dataset.

```python
test_predictions = self.reg.predict(self.X_test)
```

This generates predicted house prices for the testing data.

The training and testing predictions are returned separately through the `train()` and `test()` methods.

## 📈 10. Model Evaluation

A custom `accuracy_loss()` method is implemented to evaluate the model.

The evaluation is performed using two calculations:

### A. Accuracy Calculation

The code calculates an accuracy-like score using:

```python
acc = 1 - (n / d)
```

Where:

- `n` represents the sum of squared prediction errors.
- `d` represents the sum of squared differences between actual values and the mean of predicted values.

The calculated value is printed using:

```python
print(f"Accuracy:{acc}")
```

### B. Loss Calculation

Root Mean Squared Error (RMSE) is calculated to measure prediction error.

```python
loss = np.sqrt(n / len(y))
```

The loss is displayed using:

```python
print(f"Loss:{loss}")
```

A lower RMSE indicates that the predicted prices are closer to the actual prices.

### Evaluation Process

The model is evaluated separately on:

- Training data using `y_train` and `train_predictions`.
- Testing data using `y_test` and `test_predictions`.

```python
obj.accuracy_loss(obj.y_train, train_predictions)

obj.accuracy_loss(obj.y_test, test_predictions)
```

## 💾 11. Saving the Trained Model

After training and evaluation, the trained Linear Regression model is saved using Pickle.

```python
with open("Model.pkl", "wb") as f:
    pickle.dump(self.reg, f)
```

The trained model is stored in:

```text
Model.pkl
```

This allows the Flask application to load and use the trained model without training it again every time.

## 🌐 12. Flask Web Application

The trained Machine Learning model is integrated with Flask to create a house price prediction web application.

### Loading the Saved Model

The saved model is loaded using Pickle in `app.py`.

```python
with open("Model.pkl", "rb") as f:
    m = pickle.load(f)
```

### Creating Flask Application

```python
app = Flask(__name__)
```

### Home Page Route

The home route displays the HTML page.

```python
@app.route("/")
def home():
    return render_template("index.html")
```

### Prediction Route

The prediction route receives property details from the HTML form.

```python
@app.route("/predict", methods=["GET", "POST"])
def predict():
    a = [float(i) for i in request.form.values()]

    r = m.predict([a])

    return render_template(
        "index.html",
        prediction=float(r[0])
    )
```

**Working Process:**

1. User enters the required property details.
2. Flask receives the form values.
3. The values are converted into floating-point numbers.
4. The input values are passed to the saved model.
5. The model predicts the estimated house price.
6. The predicted price is passed back to the HTML page.

## 🔄 Complete Project Workflow

```text
             Dataset (data.csv)
                     |
                     ↓
             Data Preprocessing
                     |
                     ↓
           Remove date and street
                     |
                     ↓
             Categorical Encoding
                     |
                     ↓
       Independent and Dependent Data
                     |
                     ↓
              Train-Test Split
                     |
                     ↓
          Multiple Linear Regression
                     |
                     ↓
               Model Training
                     |
                     ↓
                Model Testing
                     |
                     ↓
              Model Evaluation
                     |
                     ↓
              Save Model.pkl
                     |
                     ↓
             Flask Web Application
                     |
                     ↓
             User Enters Details
                     |
                     ↓
             House Price Prediction
                     |
                     ↓
             Display Predicted Price
```

## 🚀 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project Folder

```bash
cd House-Price-Prediction
```

### Step 3: Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn flask
```

### Step 4: Train the Machine Learning Model

Run the following command:

```bash
python main.py
```

This performs:

- Dataset loading
- Data preprocessing
- Categorical encoding
- Training and testing data splitting
- Model training
- Model testing
- Accuracy and loss calculations
- Saving the trained model as `Model.pkl`

### Step 5: Run the Flask Application

```bash
python app.py
```

### Step 6: Open the Application

Open the following URL in your browser:

```text
http://127.0.0.1:5000/
```

Enter the property details and submit the form to get the predicted house price.

## 🔮 Future Enhancements

- Improve prediction performance.
- Experiment with other regression algorithms.
- Improve categorical encoding techniques.
- Add data visualization.
- Add input validation.
- Improve the web application interface.
- Deploy the application on a cloud platform.

## 🎓 Learning Outcomes

Through this project, the following concepts were implemented:

- Python programming
- Object-Oriented Programming
- Pandas and NumPy
- Data preprocessing
- Categorical data encoding
- Independent and dependent variable selection
- Training and testing data splitting
- Multiple Linear Regression
- Model training and testing
- Prediction and evaluation
- RMSE calculation
- Pickle model serialization
- Flask web application development
- Machine Learning model integration

## 👩‍💻 Author

**P. Kavya Sri**

Computer Science and Engineering Graduate




