# MLOps-and-Cloud-Native-AI-ML

## Mental Health Prediction Project

### Overview
This project aims to develop a robust machine learning model for predicting depression in students based on various mental health-related features. The workflow encompasses comprehensive data preprocessing, training multiple machine learning models, and creating Flask and FastAPI applications to consume the trained model via REST APIs. Additionally, the best-performing model is saved in ONNX format, and its preprocessing transformations are stored in a pickle file.

### Project Structure

#### FastAPI
- **templates:** HTML templates for the FastAPI application.
- **app.py:** FastAPI application script defining API endpoints and integrating the machine learning model.
- **Dockerfile:** Configuration for packaging the FastAPI application.
- **Procfile:** For Heroku deployment.
- **random_forest_model.onnx:** Best-performing machine learning model saved in ONNX format.
- **requirements.txt:** Text file listing required Python packages.
- **sklearn_conf_matrix.png:** Image depicting the confusion matrix generated during model evaluation.

#### Flask
- **mlruns:** Directory storing MLflow logs and experiment details.
- **templates:** HTML templates for the Flask application.
- **flask_app.py:** Flask application script creating a web interface to interact with the model.
- **Dockerfile:** Configuration for packaging the Flask application.
- **MLFlow.ipynb:** Jupyter notebook documenting the MLflow experiment, including model training runs and metrics.
- **preprocessing_steps.pkl:** Preprocessing steps saved in a pickle file.
- **preprocessing_steps_rf.pkl:** Preprocessing steps specifically for the best-performing model.
- **random_forest_model.onnx:** Best-performing machine learning model saved in ONNX format.
- **requirements.txt:** Text file listing required Python packages.
- **sklearn_conf_matrix.png:** Image showing the confusion matrix generated during model evaluation.
- **StudentMentalhealth.csv:** Dataset used for training the machine learning model.

### How to Run
#### FastAPI
- Navigate to the FastAPI directory.
- Build the Docker image: `docker build -t fastapi-app .`
- Run the Docker container: `docker run -p 80:80 fastapi-app`

#### Flask
- Navigate to the Flask directory.
- Build the Docker image: `docker build -t flask-app .`
- Run the Docker container: `docker run -p 5000:5000 flask-app`

### MLflow Experiment
- MLflow experiment details, including model training runs, metrics, and artifacts, are documented in the MLFlow.ipynb Jupyter notebook.

### Postman
- Test the APIs using Postman. Ensure correct endpoints are used for both FastAPI and Flask applications.



