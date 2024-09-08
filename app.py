from flask import Flask, render_template, request
import pickle
import pandas as pd
import shap

app = Flask(__name__)

# Load the pickled LightGBM model
with open('lightgbm_model.pkl', 'rb') as model_file:
    bst = pickle.load(model_file)
# Load the pickled encoders
with open('encoders.pkl', 'rb') as encoders_file:
    encoders = pickle.load(encoders_file)

# Load the pickled scaler
with open('minmax_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)    

# Load the pickled test data for feature scaling and encoding reference
with open('X_test.pkl', 'rb') as test_data_file:
    X_test = pickle.load(test_data_file)

# Initialize the SHAP TreeExplainer
explainer = shap.TreeExplainer(bst)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture input from the form
    form_data = request.form

    # Create a DataFrame with the input data
    new_df = pd.DataFrame({
        'Sex': [int(form_data['Sex'])],
        'GeneralHealth': [int(form_data['GeneralHealth'])],
        'SleepHours': [int(form_data['SleepHours'])],
        'RemovedTeeth': [int(form_data['RemovedTeeth'])],
        'HadAngina': [int(form_data['HadAngina'])],
        'HadStroke': [int(form_data['HadStroke'])],
        'HadAsthma': [int(form_data['HadAsthma'])],
        'HadSkinCancer': [int(form_data['HadSkinCancer'])],
        'HadCOPD': [int(form_data['HadCOPD'])],
        'HadDepressiveDisorder': [int(form_data['HadDepressiveDisorder'])],
        'HadArthritis': [int(form_data['HadArthritis'])],
        'DifficultyWalking': [int(form_data['DifficultyWalking'])],
        'ChestScan': [int(form_data['ChestScan'])],
        'AgeCategory': [int(form_data['AgeCategory'])],
        'BMI': [float(form_data['BMI'])],
        'AlcoholDrinkers': [int(form_data['AlcoholDrinkers'])],
        'HIVTesting': [int(form_data['HIVTesting'])],
        'FluVaxLast12': [int(form_data['FluVaxLast12'])],
        'PneumoVaxEver': [int(form_data['PneumoVaxEver'])],
        'CovidPos': [int(form_data['CovidPos'])],
        'HadDiabetes_No': [int(form_data['HadDiabetes_No'])],
        'SmokerStatus_Former_smoker': [int(form_data['SmokerStatus_Former_smoker'])],
        'TetanusLast10Tdap_No_did_not_receive_any_tetanus_shot_in_the_past_10_years': [int(form_data['TetanusLast10Tdap_No_did_not_receive_any_tetanus_shot_in_the_past_10_years'])],
        'TetanusLast10Tdap_Yes_received_Tdap': [int(form_data['TetanusLast10Tdap_Yes_received_Tdap'])]
    })

    # Ensure new_df has the same columns as X_test
    new_df = new_df[X_test.columns]

    # Make predictions
    y_pred_proba = bst.predict(new_df)
    y_pred_binary = [1 if x >= 0.5 else 0 for x in y_pred_proba]

    # SHAP values for the prediction
    shap_values = explainer.shap_values(new_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Generate SHAP summary
    instance_shap_values = shap_values[0]

    # Combine feature names with their SHAP values
    feature_contributions = pd.DataFrame({
        'Feature': new_df.columns,
        'SHAP Value': instance_shap_values
    }).sort_values(by='SHAP Value', ascending=False)

    # Prepare results for the frontend
    top_features = feature_contributions.head(5).to_dict(orient='records')

    # Determine the prediction result
    result = 'High Risk of Heart Attack' if y_pred_binary[0] == 1 else 'No Risk of Heart Attack'

    return render_template('result.html', prediction=result, top_features=top_features)

if __name__ == '__main__':
    app.run(debug=True)
