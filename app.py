import sys
import os

from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for a Home Page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('SuccessSage.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pref_df = data.get_data_as_data_frame()

        print(pref_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pref_df)

        return render_template('SuccessSage.html', results=round(result[0]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
