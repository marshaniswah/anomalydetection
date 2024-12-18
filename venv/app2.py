# from flask import Flask, jsonify
# from supabase import create_client
# import pandas as pd
# from pycaret.anomaly import setup, create_model, predict_model
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)

# # Supabase configuration
# supabase_url = os.getenv('SUPABASE_URL')
# supabase_key = os.getenv('SUPABASE_KEY')
# supabase = create_client(supabase_url, supabase_key)

# # Set the anomaly score threshold
# ANOMALY_SCORE_THRESHOLD = 50.0

# @app.route('/detect_anomalies', methods=['GET'])
# def detect_anomalies():
#     try:
#         # Fetch all data points from Supabase
#         response = supabase.table('esp32_1') \
#             .select('id', 'created_at', 'ds18b20_temp1', 'dht22_temp', 'ds18b20_temp2', 'ds18b20_temp3', 'ds18b20_temp4') \
#             .order('created_at', desc=True) \
#             .execute()
        
#         # Convert to DataFrame
#         df = pd.DataFrame(response.data)
        
#         # Check if 'created_at' column exists
#         if 'created_at' not in df.columns:
#             return jsonify({"error": "'created_at' column not found in the data"}), 400
        
#         # Sort by created_at to ensure correct order (newest first)
#         df = df.sort_values('created_at', ascending=False)
        
#         # Use the newest 2000 data points for anomaly detection
#         detect_data = df.head(2000)
        
#         # Remove non-numeric columns for model training
#         numeric_columns = detect_data.select_dtypes(include=['float64', 'int64']).columns
#         model_data = detect_data[numeric_columns]
        
#         try:
#             # Setup the model
#             setup(data=model_data)
            
#             # Create the model
#             model = create_model('knn')  # Using Isolation Forest algorithm
            
#             # Detect anomalies using the created model
#             predictions = predict_model(model, data=model_data)
            
#             # Add 'id' and 'created_at' back to predictions
#             predictions = pd.concat([detect_data[['id', 'created_at']], predictions], axis=1)
#         except Exception as model_error:
#             return jsonify({"error": f"Error in model creation or prediction: {str(model_error)}"}), 500
        
#         # Add 'anomaly' column based on Anomaly_Score threshold
#         predictions['anomaly'] = (predictions['Anomaly_Score'] > ANOMALY_SCORE_THRESHOLD).astype(int)
        
#         # Prepare data for Supabase update
#         update_data = predictions[['id', 'created_at', 'anomaly', 'Anomaly_Score']].rename(columns={'Anomaly_Score': 'anomaly_score'})
        
#         # Insert or update Supabase 'anomaly' table
#         for _, row in update_data.iterrows():
#             supabase.table('anomaly').upsert({
#                 'id': row['id'],
#                 'created_at': row['created_at'],
#                 'anomaly': int(row['anomaly']),
#                 'anomaly_score': float(row['anomaly_score'])
#             }).execute()
        
#         # Fetch all updated data from Supabase 'anomaly' table
#         updated_response = supabase.table('anomaly') \
#             .select('id', 'created_at', 'anomaly', 'anomaly_score') \
#             .order('created_at', desc=True) \
#             .execute()
        
#         # Convert updated data to dictionary for JSON response
#         result = pd.DataFrame(updated_response.data).to_dict(orient='records')
        
#         return jsonify(result)
    
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, jsonify
from supabase import create_client
import pandas as pd
from pycaret.anomaly import setup, create_model, predict_model
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Supabase configuration
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

# Set the anomaly score threshold
ANOMALY_SCORE_THRESHOLD = 50.0

def create_knn_model(data):
    # Setup the environment for anomaly detection
    setup_model = setup(data=data, silent=True, numeric_features=['ds18b20_temp1', 'dht22_temp', 'ds18b20_temp2', 'ds18b20_temp3', 'ds18b20_temp4'])
    
    # Create KNN model
    knn_model = create_model('knn')
    
    return knn_model

@app.route('/detect_anomalies', methods=['GET'])
def detect_anomalies():
    try:
        # Fetch all data points from Supabase
        response = supabase.table('esp32_1') \
            .select('id', 'created_at', 'ds18b20_temp1', 'dht22_temp', 'ds18b20_temp2', 'ds18b20_temp3', 'ds18b20_temp4') \
            .order('created_at', desc=True) \
            .execute()
        
        # Convert to DataFrame
        df = pd.DataFrame(response.data)
        
        # Sort by created_at to ensure correct order (newest first)
        df = df.sort_values('created_at', ascending=False)
        
        # Use the newest 2000 data points for anomaly detection
        detect_data = df.head(2000)
        
        # Create and train the KNN model
        model = create_knn_model(detect_data)
        
        # Detect anomalies using the created model
        predictions = predict_model(model, data=detect_data)
        
        # Add 'anomaly' column based on Anomaly_Score threshold
        predictions['anomaly'] = (predictions['Anomaly_Score'] > ANOMALY_SCORE_THRESHOLD).astype(int)
        
        # Prepare data for Supabase update
        update_data = predictions[['id', 'created_at', 'anomaly', 'Anomaly_Score']].rename(columns={'Anomaly_Score': 'anomaly_score'})
        
        # Insert or update Supabase 'anomaly' table
        for _, row in update_data.iterrows():
            supabase.table('anomaly').upsert({
                'id': row['id'],
                'created_at': row['created_at'],
                'anomaly': int(row['anomaly']),
                'anomaly_score': float(row['anomaly_score'])
            }).execute()
        
        # Fetch all updated data from Supabase 'anomaly' table
        updated_response = supabase.table('anomaly') \
            .select('id', 'created_at', 'anomaly', 'anomaly_score') \
            .order('created_at', desc=True) \
            .execute()
        
        # Convert updated data to dictionary for JSON response
        result = pd.DataFrame(updated_response.data).to_dict(orient='records')
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

