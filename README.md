####  Airbnb Price Prediction & Analytics

An end-to-end Machine Learning case study to predict the price (`log_price`) of Airbnb listings based on listing attributes (e.g., location, room type, listing features, reviews). The repository features an interactive web application built with Streamlit and a fully structured preprocessing and training pipeline.

###  Problem Statement
Airbnb is an online marketplace for lodging and tourism. Price is a key factor for customers when booking. The goal of this project is to build a robust Machine Learning model to predict the price (`log_price`) of a new Airbnb property based on listing attributes like room type, city, accommodations, and reviews.

---

##  Objectives
1. Perform Exploratory Data Analysis (EDA) to find pricing trends.
2. Clean and preprocess highly categorical and missing real-world data.
3. Implement and compare **Simple Linear Regression** vs **Multiple Linear Regression**.
4. Deploy an interactive **Streamlit Dashboard** for real-time price estimation.

---

## 📊 Model Performance

| Model | $R^2$ Score | MAE |
| :--- | :--- | :--- |
| **Simple Linear Regression** | 0.3155 | 0.5480 |
| **Multiple Linear Regression** | 0.5049 | 0.4041 |

*Conclusion*: The Multiple Linear Regression model provides a significantly more accurate estimation by leveraging property-specific features and location data.

---

## 📂 Project Structure

```text
├── data/                   # Dataset & Data Dictionary (AirBNB.csv, Data-Dictionary.xlsx)
├── notebook/               # Jupyter Notebook for interactive research
├── plots/                  # Automated Exploratory Data plots (distributions, correlations, etc.)
├── src/            
│   ├── airbnb_analysis.py  # Main ML training pipeline script
│   ├── streamlit_app.py    # Streamlit dashboard code
│   └── model.pkl           # Saved pre-trained Scikit-Learn model pipeline
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate to Repository
Ensure you are in the project root directory:
```bash
git clone https://github.com/Gittyasn/airbnb.git
cd airbnb
```

### 2. Set Up Virtual Environment (Optional but Recommended)
Using Python virtual environments keeps dependencies isolated:

#### Windows (Command Prompt)
```cmd
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate.bat
```

#### Windows (PowerShell)
```powershell
# Create virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1
```

#### macOS / Linux
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate
```

### 3. Install Dependencies
Install all required libraries inside the activated environment:
```bash
python -m pip install -r requirements.txt
```

---

## 📈 Training Pipeline & Modeling

To clean the data, train the regression model, and export the validation figures, run:
```bash
python src/airbnb_analysis.py
```

### What happens under the hood?
1.  **Data Loading & Cleaning**: Load `data/AirBNB.csv`, drop helper identifiers, handle missing numerical properties (using `SimpleImputer` with `median`), and sanitize categorical properties.
2.  **Exploratory Data Analysis (EDA)**: Automates generation of plots showing price distribution, room types, outliers, and feature correlations. Saved to `plots/`.
3.  **Pipeline Engineering**: Build a Scikit-Learn `Pipeline` combining:
    *   `ColumnTransformer` to route columns correctly.
    *   `StandardScaler` and `SimpleImputer` for numerical features.
    *   `OneHotEncoder(handle_unknown='ignore')` for categorical features.
    *   `LinearRegression()` as the core estimator.
4.  **Model Evaluation**: Splits data (80% train, 20% test) and prints $R^2$ score and Mean Absolute Error (MAE) comparisons.
5.  **Serialization**: Saves the fitted pipeline into `src/model.pkl` for fast inference.

---

## 💻 Running the Streamlit App

The interactive UI allows users to input listing attributes (like room type, city, number of guests, review ratings) and instantly receive a price estimation.

To run the Streamlit dashboard:

#### Option A: With an activated virtual environment
```bash
streamlit run src/streamlit_app.py
```

#### Option B: Directly via the virtual environment's Python (no activation needed)
```bash
.venv\Scripts\python -m streamlit run src/streamlit_app.py
```

### Features of the Web App:
*   **Home & Predictor Tab**: Dropdown controls, sliders, and buttons to calculate the estimated price (per night) using the serialized pipeline model.
*   **Data Analysis Tab**: Dynamic renderings of the price distributions, room type distributions, and numeric relationship graphs.
*   **About Tab**: Details listing characteristics and technical framework information.

---

## 🎨 Visualizations Overview

The pipeline saves high-quality EDA charts into the `plots/` folder:
*   `log_price_distribution.png`: Log-transformed price frequency visualization.
*   `room_type_count.png`: Frequency breakdown of available room types (Entire home/apt, Private room, Shared room).
*   `correlation_heatmap.png`: Correlation matrix of key numeric features.
*   `actual_vs_predicted.png`: Scatter plot comparing ground truth labels against model-predicted values.

---

## 🛠️ Built With

*   **Python** (Pandas, Numpy, Matplotlib, Seaborn)
*   **Scikit-Learn** (Preprocessing Pipelines, Imputers, Encoders, Linear Regression)
*   **Streamlit** (UI/UX Web Dashboard Deployment)
*   **Joblib** (Model persistence/serialization)
