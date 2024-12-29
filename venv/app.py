from flask import Flask, jsonify
from supabase import create_client
import pandas as pd
from pycaret.anomaly import load_model, predict_model
import os
from flask_cors import CORS

app = Flask(__name__)  # Fixed: Changed _name to _name_
CORS(app)

# Supabase configuration
supabase_url = 'https://uoaqpcsbemezoyfvrpbf.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVvYXFwY3NiZW1lem95ZnZycGJmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjY5MzE4OTAsImV4cCI6MjA0MjUwNzg5MH0.nD_iWXbvqxQ5xKI23265K6jwXvApLeTGEmVsNCo3zb0'
supabase = create_client(supabase_url, supabase_key)

# Load the pre-trained Isolation Forest model
model = load_model('iforest_model')

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
        
        # Detect anomalies using the pre-trained Isolation Forest model
        predictions = predict_model(model, data=detect_data)
        
        # Prepare data for Supabase update
        update_data = predictions[['id', 'created_at', 'Anomaly', 'Anomaly_Score']].rename(columns={'Anomaly': 'anomaly', 'Anomaly_Score': 'anomaly_score'})
        
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
        print(f"Error: {str(e)}")  # Added: Print the error for debugging
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':  # Fixed: Changed _name to _name_
    app.run(debug=True)
