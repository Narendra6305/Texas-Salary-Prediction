# 🧾 Texas Salary Prediction

A machine learning project focused on predicting the annual salary of Texas state employees using structured government payroll data. The project includes data cleaning, exploratory analysis, feature engineering, model training, and performance evaluation.

---

## 📌 Project Description

This project aims to forecast employee salaries in the Texas state government workforce using various features such as job class, department (agency), demographic details, and compensation structure.

With over 100,000 records across multiple agencies, this dataset provides insight into wage disparities, staffing structures, and compensation policies. A regression model is trained to estimate the `Annual Salary` of employees based on meaningful input features.

**Goals:**

- Predict employee salaries based on existing HR records.
- Compare machine learning models to identify the best performer.
- Provide a baseline for government or organizational salary benchmarking tools.

---

## 📊 Features Used

- **Job Class**
- **Agency Name**
- **Weekly Hours**
- **Hourly Rate**
- **Employment Status (Full-time/Part-time)**
- **Gender, Ethnicity**
- **Annual Salary (target)**

---

## 🔧 Technologies & Libraries

- Python
- Pandas & NumPy
- Scikit-learn
- Seaborn & Matplotlib
- Jupyter Notebook

---

## 🔎 Workflow

### 1. Data Preprocessing
- Remove irrelevant columns (e.g., names, IDs)
- Handle missing values
- Encode categorical features (e.g., agency, class title)
- Normalize or standardize numeric columns

### 2. Exploratory Data Analysis (EDA)
- Visualize distributions of annual salary
- Compare average salary across agencies, job roles, gender, and employment status
- Identify and handle outliers

### 3. Model Training
- Train multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor

### 4. Evaluation Metrics
- R² Score (Coefficient of Determination)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Adjusted R² (if applicable)

---

## 🧠 Sample Code Snippet

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Prepare features and labels
X = df.drop(columns=['Annual_Salary'])
y = df['Annual_Salary']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

📁 Project Structure
├── data/
│   └── texas_salary_data.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   └── Texas Salary Prediction.ipynb
├── requirements.txt
└── README.md

📈 Key Insights
Job class and agency are among the strongest predictors of salary.

Full-time employees generally earn significantly more than part-time staff.

Linear Regression gave competitive results with good interpretability.

Outliers (extremely high salaries) were capped or removed to improve model accuracy.

🚀 Future Enhancements
Add support for time-series trend forecasting.

Hyperparameter tuning for boosting models.

Deploy the model with a Streamlit dashboard for interactive use.

Use SHAP or LIME for feature impact explanation.


✨ Credits
Developed by Narendra6305
Inspired by public Texas state payroll datasets and machine learning regression best practices.

---

Let me know if you'd like a version with visuals (e.g., embedded images or charts), or if you're deploying the model (so I can help write an API or app section).

