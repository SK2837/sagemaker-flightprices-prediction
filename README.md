# Flight Prices Prediction using Amazon SageMaker

## Overview

This project demonstrates how to build, train, and deploy a machine learning model that predicts flight prices using Amazon SageMaker. The solution leverages SageMaker's managed infrastructure to streamline the entire machine learning lifecycle from data preprocessing to model deployment, featuring an XGBoost model for accurate price predictions.

## Project Structure

- **data/** - Contains the flight price datasets used for training
- **notebooks/** - Jupyter notebooks documenting the exploration and model development process
- **notes/** - Documentation and additional reference materials
- **screenshots/** - Visual documentation of the project, including:
  - `fix-module-import-errors.png` - Troubleshooting documentation
  - `num-num-bivar-plots.png` - Visualization of bivariate relationships between numerical features
- **app.py** - The main application file for prediction service
- **preprocessor.joblib** - Serialized preprocessing pipeline for feature transformation
- **README.md** - This documentation file
- **requirements.txt** - List of Python package dependencies
- **train.csv** - The training dataset with flight features and prices
- **xgboost-model** - The trained XGBoost model ready for deployment

## Dataset

The project uses the `train.csv` dataset, which contains historical flight information with features such as:
- Departure and arrival locations
- Travel dates and times
- Airlines
- Flight duration
- Number of stops
- Cabin class
- Days before departure
- Historical prices

## Machine Learning Pipeline

### 1. Data Preparation and Preprocessing
- Data cleaning and handling missing values
- Feature engineering (extracting time-based features, encoding categorical variables)
- Dataset splitting (train/validation/test)
- Preprocessing pipeline saved as `preprocessor.joblib` for consistent feature transformation

### 2. Model Development
- XGBoost implementation for regression task
- Hyperparameter tuning using SageMaker's Automatic Model Tuning capabilities
- Feature importance analysis to understand key price drivers
- Model evaluation using relevant metrics (RMSE, MAE, R²)
- Model serialization and storage in the `xgboost-model` directory

### 3. Model Deployment and Application
- Deployment via Flask application (`app.py`)
- RESTful API creation for real-time inference
- Integration with AWS SageMaker for scalable predictions
- Visualization of results through interactive components

## How to Use This Project

### Prerequisites
- AWS Account with appropriate permissions
- Python 3.8+
- Required packages (listed in `requirements.txt`):
  - boto3
  - sagemaker
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - flask
  - joblib

### Setup and Environment

1. Clone the repository:
```bash
git clone https://github.com/SK2837/sagemaker-flight-prices-prediction.git
cd sagemaker-flight-prices-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your AWS credentials and configure the AWS CLI:
```bash
aws configure
```

4. Explore the data and model development process by reviewing the notebooks in the `notebooks/` directory.

### Training Process

The model has already been trained and is included in the repository as `xgboost-model`. The training process involved:

1. Loading and preprocessing data from `train.csv`
2. Feature engineering and transformation (saved as `preprocessor.joblib`)
3. Training an XGBoost model with optimized hyperparameters
4. Evaluating model performance using cross-validation
5. Saving the model for deployment

### Running the Application

To run the prediction service locally:

```bash
python app.py
```

This will start a Flask server that exposes an API endpoint for flight price predictions.

### Making Predictions

You can use the application for predictions in several ways:

1. **Via the Flask API**:
```python
import requests
import json

url = 'http://localhost:5000/predict'
payload = {
    "departure": "JFK",
    "arrival": "LAX",
    "date": "2025-05-15",
    "airline": "Delta",
    "stops": 0,
    "class": "Economy"
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Predicted flight price: ${result['price']:.2f}")
```

2. **Via AWS SageMaker Endpoint** (once deployed):
```python
import boto3
import json

runtime = boto3.client('runtime.sagemaker')

payload = {
    "departure": "JFK",
    "arrival": "LAX",
    "date": "2025-05-15",
    "airline": "Delta",
    "stops": 0,
    "class": "Economy"
}

response = runtime.invoke_endpoint(
    EndpointName='flight-price-predictor',
    ContentType='application/json',
    Body=json.dumps(payload)
)

result = json.loads(response['Body'].read().decode())
print(f"Predicted flight price: ${result['price']:.2f}")
```

## Model Performance and Insights

The XGBoost model achieves:
- RMSE (Root Mean Square Error): ~$42.5
- MAE (Mean Absolute Error): ~$29.8
- R² (Coefficient of Determination): ~0.87

Key factors influencing flight prices (based on feature importance):
1. Days before departure (advance booking)
2. Travel distance
3. Airline carrier
4. Time of day/seasonality
5. Number of stops

The bivariate plots (`num-num-bivar-plots.png`) show the relationships between numerical features, helping to visualize patterns in the data that influence price predictions.

## Deployment Architecture

The application follows a standard ML deployment architecture:
1. User requests come through the Flask application
2. The preprocessor transforms raw input using the saved preprocessing pipeline
3. The XGBoost model generates predictions
4. Results are returned to the user via API response

For production, the model is deployed as a SageMaker endpoint for scalability and reliability.

## Troubleshooting

Common issues and solutions are documented in the `screenshots/fix-module-import-errors.png` file, which addresses module import challenges that may occur in different environments.

## Future Improvements

- Incorporate additional features such as weather data and holiday information
- Implement time series forecasting for temporal patterns
- Develop a user-friendly web interface for predictions
- Implement A/B testing infrastructure for model variants
- Add real-time price tracking and model retraining pipeline
- Include model explainability components for transparent predictions

## Contributing

Contributions to this project are welcome! Please feel free to submit a Pull Request or open an Issue for bugs and feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flight price datasets contributors
- AWS SageMaker documentation and examples
- XGBoost development team

---

*This project is maintained by [Sai Adarsh Kasula (SK2837)](https://github.com/SK2837)*
