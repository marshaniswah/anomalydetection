# IoT Anomaly Detection with Flask and Supabase

This project is a Flask-based API for detecting anomalies in IoT sensor data using a pre-trained Isolation Forest model. The application integrates with Supabase for data storage and retrieval and uses PyCaret for anomaly detection.

---

## Features

- Fetches IoT sensor data from Supabase.
- Processes and detects anomalies using a pre-trained Isolation Forest model.
- Updates the results back into Supabase for storage and visualization.
- Provides a REST API for accessing the anomaly detection results.
- Supports Cross-Origin Resource Sharing (CORS).

---

## Tech Stack

- **Python**: Flask, PyCaret, Pandas
- **Database**: Supabase
- **Model**: Isolation Forest
- **Others**: Flask-CORS, dotenv

---

## Setup Instructions

### Prerequisites
- Python 3.8 or later
- Supabase account with a configured database
- A pre-trained Isolation Forest model (`iforestmodel.pkl`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install required dependencies:
```bash
 pip install -r requirements.txt
```
3. Create a .env file and add your Supabase configuration:
```bash
  SUPABASE_URL=your_supabase_url
  SUPABASE_KEY=your_supabase_key
```
4. Place the pre-trained model file (iforestmodel.pkl) in the project directory.

### Usage
1. Start the Flask application:
```bash
  python app.py
```
2. Access the API endpoint:
```bash
  GET http://127.0.0.1:5000/detect_anomalies
```

3. View results in your Supabase database or the JSON response.

#### API Endpoint
GET /detect_anomalies
Description: Detects anomalies in the latest IoT sensor data.
Response: JSON with anomaly detection results, including:
id: Record ID
created_at: Timestamp
anomaly: 1 (anomalous) or 0 (normal)
anomaly_score: Anomaly score
