from flask import Flask, render_template, request, jsonify
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# Load the dataset
file_path = 'newdata.csv'
data2 = pd.read_csv(file_path)

# Convert 'outbreak_starting_date' to datetime
data2['outbreak_starting_date'] = pd.to_datetime(data2['outbreak_starting_date'], errors='coerce')

# Ensure 'cases' column is numeric
data2['cases'] = pd.to_numeric(data2['cases'], errors='coerce')

# Drop rows with missing values
data2 = data2.dropna(subset=['outbreak_starting_date', 'cases', 'disease_illness_name', 'state', 'Season'])

@app.route('/')
def home():
    return render_template('app.html')

@app.route('/state-wise')
def state():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    state = request.form['state']
    season = request.form['season']

    # Filter data for the specific state and season
    filtered_data = data2[(data2['state'] == state) & (data2['Season'] == season)]

    # Aggregate cases by disease
    disease_data = filtered_data.groupby('disease_illness_name').agg(
        total_cases=('cases', 'sum'),
        frequency=('cases', 'count')
    ).reset_index()

    # Calculate historical probabilities for each disease
    total_cases = disease_data['total_cases'].sum()
    disease_data['probability'] = disease_data['total_cases'] / total_cases

    # ARIMA Forecast for each disease
    forecasted_diseases = {}

    for _, row in disease_data.iterrows():
        disease = row['disease_illness_name']
        disease_history = filtered_data[filtered_data['disease_illness_name'] == disease]
        disease_history = disease_history.groupby('outbreak_starting_date')['cases'].sum().reset_index()
        disease_history.set_index('outbreak_starting_date', inplace=True)
        
        # Ensure continuous time series
        disease_history = disease_history.asfreq('D', fill_value=0)
        
        try:
            # Train ARIMA model
            model = ARIMA(disease_history['cases'], order=(5, 1, 0))
            model_fit = model.fit()
            
            # Forecast next 10 steps
            forecast = model_fit.forecast(steps=10)
            forecasted_diseases[disease] = forecast.sum() * row['probability']
        except Exception as e:
            print(f"Error forecasting for {disease}: {e}")
            forecasted_diseases[disease] = row['total_cases'] * row['probability']  # Fallback to historical cases

    # Sort the results by the number of predicted cases
    sorted_predictions = sorted(forecasted_diseases.items(), key=lambda x: x[1], reverse=True)

    # Render results to the frontend
    return render_template('results.html', state=state, season=season, predictions=sorted_predictions)

import io
import base64
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

@app.route('/all-states', methods=['GET', 'POST'])
def all():
    if request.method == 'POST':
        # Get user input
        region = request.form['region']
        season = request.form['season']
        threshold = int(request.form['threshold'])  # Minimum case threshold
        
        # Define regions and states
        region_states = {
            'Central': ['Madhya Pradesh', 'Chhattisgarh', 'Jharkhand', 'Bihar'],
            'South': ['Tamil Nadu', 'Karnataka', 'Kerala', 'Andhra Pradesh', 'Telangana'],
            'North': ['Uttar Pradesh', 'Himachal Pradesh', 'Punjab', 'Harayana', 'Jammu & Kashmir'],
            'West': ['Rajasthan', 'Gujarat', 'Maharashtra', 'Goa'],
            'East': ['West Bengal', 'Odisha', 'Mizoram', 'Manipur', 'Nagaland', 'Arunachal Pradesh', 'Meghalaya', 'Tripura', 'Assam', 'Sikkim']
        }
        
        # Filter data based on region and season
        selected_states = region_states.get(region, [])
        filtered_data = data2[(data2['state'].isin(selected_states)) & (data2['Season'] == season)]
        
        # Aggregate cases by disease and filter by threshold
        disease_data = filtered_data.groupby('disease_illness_name').agg(
            total_cases=('cases', 'sum')
        ).reset_index()
        disease_data = disease_data[disease_data['total_cases'] > threshold]  # Apply threshold filter

        # Generate bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(disease_data['disease_illness_name'], disease_data['total_cases'], color='skyblue')
        plt.xlabel('Disease', fontsize=14)
        plt.ylabel('Number of Cases', fontsize=14)
        plt.title(f'Diseases in {region} Region during {season} (Cases > {threshold})', fontsize=16)
        plt.xticks(rotation=45, ha='right')

        # Save chart as a base64 string
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Precautions based on season
        precautions = {
            'Summer': ['Stay hydrated', 'Avoid direct sunlight', 'Wear light clothing'],
            'Monsoon': ['Use mosquito repellents', 'Drink clean water', 'Avoid waterlogged areas'],
            'Winter': ['Stay warm', 'Avoid cold drinks', 'Eat immune-boosting foods'],
            'Autumn': ['Avoid pollen exposure', 'Keep surroundings clean', 'Use protective masks']
        }

        return render_template(
            "allstates.html", 
            region=region, 
            season=season, 
            chart_base64=chart_base64, 
            precautions=precautions.get(season, []),
            threshold=threshold
        )
    return render_template("allstates.html")


if __name__ == '__main__':
    app.run(debug=True, port=5002)