# 🛍️ Retail Sales Prediction with Random Forest

This project builds and evaluates various **Random Forest models** to predict retail sales based on time, holiday, and store-specific features. It includes preprocessing, model training with hyperparameter tuning, evaluation, and prediction on new datasets.

## 📂 Files

- `csv_files/sales.csv` — Main dataset used for model training.
- `csv_files/ironkaggle_notarget.csv` — New data for predictions (without target).
- `csv_files/ironkaggle_solutions.csv` — Comparison/solution dataset.
- `models/` — Folder containing saved models (`.pkl` format).

## 🧠 Machine Learning Models

Three main Random Forest models were trained:

### ✅ `model_rf01`
- Features: All features except `sales`
- Normalization: `StandardScaler`
- Hyperparameters: `n_estimators=100`, `max_depth=20`
- Status: Trained and saved to `models/model_rf01.pkl`

### ✅ `model_rf02`
- Features: Dropped `open`, `is_weekend`, `school_holiday`, `year`
- Normalization: `StandardScaler`
- Hyperparameters: `n_estimators=100`, `max_depth=None`
- Status: Trained and saved to `models/model_rf02.pkl`

### ✅ `model_rf03`
- Grid Search (classification mistake — used `scoring='f1'`)
- Status: Trained, but improperly scored for regression

### ⚙️ HalvingGridSearchCV (Experimental)
- Feature engineering and time estimate logic for efficient hyperparameter search
- Output model: `models/best_rf_halving.pkl`
- Status: Time estimation and progress tracking using `tqdm` & `threading`

## 🤖 Other Models Tried

Several other machine learning models were imported (e.g., `XBoost`, `GradientBoostingRegressor`, `lightgbm` etc.) and briefly tested. However, they performed significantly worse than Random Forest on this dataset.  
As a result, we removed their implementations to keep the project clean and focused on the best-performing model.

## 📊 Evaluation Metrics

Each model is evaluated using:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R² Score**: Coefficient of Determination

## 🔍 Feature Importance

Feature importances were plotted using `model.feature_importances_` to understand which variables most influence predictions.

## 📈 Correlation Matrix

A masked heatmap using `seaborn` highlights correlations between features.

## 📤 Prediction on New Data

The file `ironkaggle_notarget.csv` was:
- Preprocessed with the same steps as training data
- Normalized using the previously fitted `StandardScaler`
- Predicted using `model_rf02`
- Output stored in `df01_with_predictions.csv`

## 🛠️ Tools Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`: RandomForestRegressor, GridSearchCV, HalvingGridSearchCV
- `joblib`: For model saving/loading
- `tqdm` and `threading`: For progress monitoring

## 📝 How to Run

1. Ensure required libraries are installed (`requirements.txt` or use `pip install -r requ

## 👥 Contributors

We welcome contributions from anyone interested in improving this project! Here are the current contributors:

* **Author**: JorgeMMLRodrigues
* **Email**: jorgemmlrodrigues@gmail.com
* **GitHub**: https://github.com/JorgeMMLRodrigues

* **Author**: Mic-dev-gif
* **Email**: michelemontalvo@outlook.com
* **GitHub**: https://github.com/Mic-dev-gif

* **Email**: simiatawane@gmail.com

* **Email**: felipe.rocha@ironhack.com
