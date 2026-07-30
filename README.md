 Airbnb Price Prediction: Machine Learning Case Study

📝 Problem Statement
Airbnb is an online marketplace for lodging and tourism. Price is a key factor for customers when booking. The goal of this project is to build a robust Machine Learning model to predict the price (log_price) of a new Airbnb property based on listing attributes like room type, city, accommodations, and reviews.

📊 Dataset Overview
The dataset contains 74,111 Airbnb listings across various cities.

Target Variable: log_price
Key Features: room_type, accommodates, bathrooms, number_of_reviews, review_scores_rating, city, instant_bookable, etc.
## Objectives
Perform Exploratory Data Analysis (EDA) to find pricing trends.
Clean and preprocess highly categorical and missing real-world data.
Implement and compare Simple Linear Regression vs Multiple Linear Regression.
Deploy an interactive Streamlit Dashboard for real-time price estimation.
# Getting Started
1. Installation
Clone the repository and install the dependencies:

bash
git clone https://github.com/Gittyasn/airbnb.git
cd airbnb
pip install -r requirements.txt
2. Run the Analysis
To execute the full data cleaning, visualization, and model training pipeline:

bash
python src/airbnb_analysis.py
3. Launch the Web Dashboard
To use the interactive price predictor:

bash
python -m streamlit run src/streamlit_app.py
# Model Performance
Model	R2 Score	MAE
Simple Linear Regression	0.3155	0.5480
Multiple Linear Regression	0.5049	0.4041
Conclusion: The Multiple Linear Regression model provides a significantly more accurate estimation by leveraging property-specific features and location data.

📁 Folder Structure
text

├── data/           # Dataset & Data Dictionary
├── notebook/       # Jupyter Notebook for interactive research
├── plots/          # Automated Exploratory Data plots
├── src/            
│   ├── airbnb_analysis.py  # Main ML script
│   ├── streamlit_app.py    # Dashboard code
│   └── model.pkl           # Saved pre-trained model
├── requirements.txt # Project dependencies
└── README.md        # Project documentation

# Built With
Python (Pandas, Numpy, Seaborn)
Scikit-Learn (Modeling & Pipelines)
Streamlit (UI/UX deployment)
Joblib (Model persistence)
