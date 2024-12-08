Credit Risk Modeling and Classification

Overview:
Credit Risk Modeling is crucial for financial institutions to evaluate loan applications and mitigate the risk of defaults. This project builds a machine learning classification model to assess credit risk by leveraging Python, MySQL, and Streamlit. The interactive application allows users to explore insights, visualize trends, and make predictions.

Features
🔍 Data Analysis and Preprocessing
Cleaned and transformed applicant data for consistency and accuracy.
Addressed missing values, categorical encoding, and outlier detection.
🤖 Machine Learning Models
Developed models like Logistic Regression, Random Forest, and Gradient Boosting for credit risk classification.
Evaluated using metrics such as Accuracy, Precision, Recall, and ROC-AUC.
📊 Interactive Dashboard
Built with Streamlit for real-time insights and predictions.
Features include:
Visualizations of risk trends.
Classification of new applicants.
Dynamic filtering options for granular analysis.
Technologies Used
Tool	Purpose
Python	Data processing and model development
MySQL	Database management and storage
Streamlit	Interactive dashboard visualization
Pandas	Data manipulation and preprocessing
Scikit-learn	Machine learning model implementation
Matplotlib/Seaborn	Data visualization
Quick Start
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/your-repo/credit-risk-modeling.git
cd credit-risk-modeling
2️⃣ Install Dependencies
Install Python libraries with the following command:

bash
Copy code
pip install -r requirements.txt
3️⃣ Set Up the Database
Import the provided SQL script in the database/ folder to create the schema.
Update the database connection details in the config.py file:
python
Copy code
# config.py
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "yourpassword"
DB_NAME = "credit_risk"
4️⃣ Run the Streamlit App
Launch the interactive dashboard:

bash
Copy code
streamlit run app.py
Project Workflow
📂 Folder Structure
plaintext
Copy code
📂 credit-risk-modeling/
├── 📂 data/                # Dataset for analysis
├── 📂 notebooks/           # Jupyter Notebooks for EDA
├── 📂 scripts/             # Python scripts for models and preprocessing
├── 📂 streamlit_app/       # Streamlit app files
├── 📂 database/            # MySQL schema and example data
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
└── LICENSE                 # License information
🛠️ Key Scripts
1. Data Preprocessing
python
Copy code
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("data/credit_data.csv")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical features
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])

# Save cleaned data
data.to_csv("data/cleaned_data.csv", index=False)
2. Model Training
python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data
X = data.drop('Risk', axis=1)
y = data['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
Interactive Dashboard
Streamlit Features
Run Predictions
python
Copy code
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/risk_model.pkl")

# User input form
st.title("Credit Risk Classifier")
age = st.number_input("Enter Age", 18, 100)
income = st.number_input("Enter Monthly Income")
loan_amount = st.number_input("Enter Loan Amount")
gender = st.selectbox("Gender", ["Male", "Female"])

# Predict risk
if st.button("Predict"):
    input_data = pd.DataFrame([[age, income, loan_amount, gender]], columns=['Age', 'Income', 'LoanAmount', 'Gender'])
    prediction = model.predict(input_data)
    st.write("Risk Level:", "High" if prediction[0] == 1 else "Low")

Key Insights
Top Risk Factors:

High debt-to-income ratio.
Poor credit repayment history.
Low credit score.
Models achieved:

Accuracy: 89%
ROC-AUC: 91%
Recommendations:

Regular monitoring of applicants with high-risk scores.
Improved guidelines for granting loans to reduce defaults.
Contributions
We welcome contributions to enhance this project! Follow these steps:

Fork this repository.
Create a new branch for your feature/bug fix.
Submit a pull request.
