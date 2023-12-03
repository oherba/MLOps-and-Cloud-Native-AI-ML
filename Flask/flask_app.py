from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('prediction.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = {
            'Gender': request.form['Gender'],
            'Age': request.form['Age'],
            'Course': request.form['Course'],
            'Year_of_Study': request.form['Year_of_Study'],
            'CGPA': request.form['CGPA'],
            'Marital_Status': request.form['Marital_Status'],
            'Anxiety': request.form['Anxiety'],
            'Panic_Atack': request.form['Panic_Atack'],
            'Treatment': request.form['Treatment'],
        }

        response = requests.post('http://localhost:5000/predict', data=data)
        prediction_text = response.json().get('prediction_text', '')

        return render_template('prediction.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
