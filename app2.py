from flask import Flask, render_template_string, request, url_for, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
from urllib.parse import unquote

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# --- File Paths ---
MODEL_FILE = 'CarPricePredictor_LR_model_1.pkl'
DATA_FILE = 'Cleaned_Car_dataset.csv'

# --- Load Model and Data ---
if not os.path.exists(MODEL_FILE) or not os.path.exists(DATA_FILE):
    print("="*80)
    print(f"FATAL ERROR: Required file not found.")
    print(f"Model file expected at: '{os.path.abspath(MODEL_FILE)}'")
    print(f"Data file expected at:  '{os.path.abspath(DATA_FILE)}'")
    print("Please ensure both files are in the same directory as this script.")
    print("="*80)
    exit()

try:
    model = pickle.load(open(MODEL_FILE, 'rb'))
    car = pd.read_csv(DATA_FILE)
    car['name'] = car['name'].str.strip()
    car['company'] = car['company'].str.strip()
    print("Model and data loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# --- Custom Jinja Filter ---
def format_number(value):
    return f"{value:,}"
app.jinja_env.filters['format_number'] = format_number

# --- HTML Templates ---
HTML_FORM = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Predictor</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='newstyle.css') }}">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="card-header">
                <h1>Car Price Predictor</h1>
                <p class="subtitle">Fill in the details below to get a price estimate.</p>
            </div>
            <form id="prediction-form" action="{{ url_for('predict') }}" method="post">
                <div class="form-group">
                    <label for="company">Company</label>
                    <select name="company" id="company" required>
                        {% for c in companies %}
                        <option value="{{ c }}">{{ c }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label for="car_models">Model</label>
                    <select name="car_models" id="car_models" required disabled>
                        <option>Select Company First</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="year">Year of Purchase</label>
                    <select name="year" id="year" required disabled>
                        <option>Select Model First</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="fuel_type">Fuel Type</label>
                    <select name="fuel_type" id="fuel_type" required>
                        {% for f in fuel_types %}
                        <option value="{{ f }}">{{ f }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="form-group">
                    <label for="kilo_driven">Kilometers Driven</label>
                    <input type="text" name="kilo_driven" id="kilo_driven" placeholder="e.g., 50000" required pattern="[0-9]+" title="Please enter numbers only."/>
                </div>
                <button type="submit" class="submit-btn">Predict Price</button>
            </form>
        </div>
        <footer class="footer">
            <p>Built with Flask & Scikit-learn</p>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const companySelect = document.getElementById('company');
            const modelSelect = document.getElementById('car_models');
            const yearSelect = document.getElementById('year');

            async function populateSelect(url, targetSelect, defaultOptionText, dataKey) {
                targetSelect.innerHTML = '<option>Loading...</option>';
                targetSelect.disabled = true;

                try {
                    const response = await fetch(url);
                    if (!response.ok) {
                        throw new Error(`Network Error: ${response.status} ${response.statusText}`);
                    }
                    const data = await response.json();

                    if (data.error) {
                        throw new Error(`Server Error: ${data.error}`);
                    }

                    targetSelect.innerHTML = '';
                    const items = data[dataKey] || [];
                    
                    const defaultOption = document.createElement('option');
                    defaultOption.textContent = defaultOptionText;
                    targetSelect.appendChild(defaultOption);

                    if (items.length > 0) {
                        items.forEach(item => {
                            const option = document.createElement('option');
                            option.value = item;
                            option.textContent = item;
                            targetSelect.appendChild(option);
                        });
                        targetSelect.disabled = false;
                    } else {
                        defaultOption.textContent = "No options available";
                        targetSelect.disabled = true;
                    }
                } catch (error) {
                    console.error(`Failed to populate ${targetSelect.id}:`, error);
                    targetSelect.innerHTML = '<option>Error loading data</option>';
                    targetSelect.disabled = true;
                }
            }
            
            companySelect.addEventListener('change', function() {
                const selectedCompany = this.value;
                yearSelect.innerHTML = '<option>Select Model First</option>';
                yearSelect.disabled = true;

                if (selectedCompany && selectedCompany !== 'Select Company') {
                    // Use encodeURIComponent for the company name as well, for robustness
                    const encodedCompany = encodeURIComponent(selectedCompany);
                    populateSelect(`/get_models/${encodedCompany}`, modelSelect, 'Select Model', 'models');
                } else {
                    modelSelect.innerHTML = '<option>Select Company First</option>';
                    modelSelect.disabled = true;
                }
            });

            modelSelect.addEventListener('change', function() {
                const selectedModel = this.value;
                if (selectedModel && selectedModel !== 'Select Model') {
                    const encodedModel = encodeURIComponent(selectedModel);
                    populateSelect(`/get_years/${encodedModel}`, yearSelect, 'Select Year', 'years');
                } else {
                    yearSelect.innerHTML = '<option>Select Model First</option>';
                    yearSelect.disabled = true;
                }
            });
        });
    </script>
</body>
</html>"""  
HTML_PREDICTION_RESULT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='newstyle.css') }}">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <div class="card result-card">
            <div class="card-header">
                <h1>Prediction Result</h1>
            </div>

            <div class="prediction-output {% if 'Error' in prediction_text %}error-message{% endif %}">
                <h2>{{ prediction_text }}</h2>
            </div>
            
            {% if 'Error' not in prediction_text %}
            <div class="details-section">
                <h3>Submitted Details</h3>
                <table class="details-table">
                    <tr>
                        <td>Company</td>
                        <td>{{ company }}</td>
                    </tr>
                    <tr>
                        <td>Model</td>
                        <td>{{ car_model }}</td>
                    </tr>
                    <tr>
                        <td>Year</td>
                        <td>{{ year }}</td>
                    </tr>
                    <tr>
                        <td>Fuel Type</td>
                        <td>{{ fuel_type }}</td>
                    </tr>
                    <tr>
                        <td>Kilometers Driven</td>
                        <td>{{ driven | int | format_number }} km</td>
                    </tr>
                </table>
            </div>
            {% endif %}

            <a href="/" class="back-link">Predict Another Car</a>
        </div>
         <footer class="footer">
            <p>Built with Flask & Scikit-learn</p>
        </footer>
    </div>
</body>
</html>"""  

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def index():
    companies = sorted(car['company'].dropna().unique())
    fuel_types = sorted(car['fuel_type'].dropna().unique())
    companies.insert(0, "Select Company")
    fuel_types.insert(0, "Select Fuel Type")
    return render_template_string(HTML_FORM, companies=companies, fuel_types=fuel_types)

@app.route('/get_models/<company>')
def get_models(company):
    company_decoded = unquote(company).strip()
    print(f"\n[API_CALL] /get_models/ for company: '{company_decoded}'")

    if not company_decoded or company_decoded == 'Select Company':
        return jsonify({'models': []})
    try:
        models = sorted(car[car['company'].str.strip() == company_decoded]['name'].unique())
        print(f"[API_RESPONSE] Found {len(models)} models: {models}")
        return jsonify({'models': models})
    except Exception as e:
        print(f"[API_ERROR] in get_models: {e}")
        return jsonify({'error': 'Error fetching models'}), 500

@app.route('/get_years/<path:car_model>')
def get_years(car_model):
    car_model_decoded = unquote(car_model).strip()
    print(f"\n[API_CALL] /get_years/ for model: '{car_model_decoded}'")

    if not car_model_decoded or car_model_decoded == 'Select Model':
        return jsonify({'years': []})

    try:
        model_data = car[car['name'].str.strip() == car_model_decoded]
        if model_data.empty:
            print(f"[API_ERROR] Model '{car_model_decoded}' not found in dataset.")
            return jsonify({'error': 'Model not found in dataset'}), 404

        min_year = int(model_data['year'].min())
        all_years = sorted(car['year'].dropna().unique().astype(int), reverse=True)
        valid_years = [int(year) for year in all_years if year >= min_year]

        print(f"[API_RESPONSE] Valid years for '{car_model_decoded}': {valid_years}")
        return jsonify({'years': valid_years})
    except Exception as e:
        print(f"[API_ERROR] in get_years: {e}")
        return jsonify({'error': 'Error fetching years'}), 500


@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")
    car_model = request.form.get("car_models")
    year_str = request.form.get("year")
    fuel_type = request.form.get("fuel_type")
    kilo_driven_str = request.form.get("kilo_driven", "")

    if (company == "Select Company" or car_model == "Select Model" or 
        year_str == "Select Year" or fuel_type == "Select Fuel Type" or
        not kilo_driven_str.isdigit()):
        return render_template_string(HTML_PREDICTION_RESULT, prediction_text="Error: Please fill out all fields correctly.")

    try:
        year = int(year_str)
        driven = int(kilo_driven_str)

        input_df = pd.DataFrame([[car_model.strip(), company.strip(), year, driven, fuel_type.strip()]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        print(f"\n[PREDICT] Input: {input_df.to_dict('records')}")
        prediction = model.predict(input_df)
        predicted_price = np.round(prediction[0], 2)
        print(f"[PREDICT] Output: ₹{predicted_price:,.2f}")

        if predicted_price < 0:
            prediction_text = "Error: Unrealistic prediction. Please check the KM Driven value."
        else:
            prediction_text = f"Estimated Price: ₹{predicted_price:,.2f}"
    except Exception as e:
        print(f"[PREDICT_ERROR] {e}")
        prediction_text = "Error: An error occurred during prediction."

    return render_template_string(
        HTML_PREDICTION_RESULT, 
        prediction_text=prediction_text,
        company=company,
        car_model=car_model,
        year=year_str,
        fuel_type=fuel_type,
        driven=kilo_driven_str
    )

if __name__ == "__main__":
    app.run(debug=True)