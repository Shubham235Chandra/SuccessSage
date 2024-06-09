# SuccessSage: Student Exam Performance Indicator and Predictor

SuccessSage is a comprehensive end-to-end machine learning project aimed at predicting student exam performances. It leverages a variety of educational and demographic data to provide insights and predictions, enabling educational stakeholders to better understand and improve student outcomes.

## Features

- **Performance Prediction**: Utilizes a variety of machine learning models to predict student performance in exams based on their demographic and academic data.
- **Model Optimization**: Employs a grid search to find the best hyperparameters for each model, ensuring optimal performance.
- **Data Pipeline**: Includes complete data ingestion, transformation, and model training pipelines, which automate the process from raw data to predictions.
- **Web Interface**: An interactive Flask-based web interface for easy data input and result visualization.
- **Logging and Error Handling**: Comprehensive logging and custom exception handling to ensure traceability and ease of debugging.

## Data Overview

The project utilizes a dataset containing several features that are indicative of student academic performance:

- **Demographic Information**: Includes gender, race/ethnicity, and whether the student receives a free/reduced lunch.
- **Academic Background**: Covers parental level of education and whether the student completed a test preparation course.
- **Scores**: Reading and writing scores are used as input features.

### Data Processing

- **Data Ingestion**: Automatically ingests data from a CSV file, splits it into training and testing datasets.
- **Data Transformation**: Transforms categorical data using one-hot encoding and scales numerical features to prepare them for model training.
- **Training and Test Datasets**: The prepared data is split into training and testing sets to evaluate model performance.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Shubham235Chandra/SuccessSage.git
   cd SuccessSage
   ```

2. **Set Up a Virtual Environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask App**
   ```bash
   python app.py
   ```
   This will run the web application on `localhost` on port `5000`.

2. **Access the Web Interface**
   - Open a browser and navigate to `http://localhost:5000/`.
   - Use the web form to input student data and receive predictions.

## Project Structure

- `app.py`: Flask application's entry point.
- `application.py`: Manages routes and web form handling.
- `predict_pipeline.py`: Manages the prediction pipeline including preprocessing and model predictions.
- `data_ingestion.py`: Manages the ingestion and initial processing of data.
- `data_transformation.py`: Implements the preprocessing pipeline.
- `model_trainer.py`: Manages the training of machine learning models using various algorithms such as RandomForest, DecisionTree, GradientBoosting, and more.
- `utils.py`: Utility functions for serialization and other tasks.
- `logger.py`: Configures logging for monitoring.
- `exception.py`: Custom exception handling for robust error management.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your proposed changes.

## License

Distributed under the MIT License. See `LICENSE` file for more information.
