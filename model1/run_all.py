import os
import subprocess

def run_data_collection():
    """Run data collection script."""
    print("Starting data collection...")
    subprocess.run(["python", "data_collection.py"])
    print("Data collection complete.")

def run_data_training():
    """Run data training script."""
    print("Starting model training...")
    subprocess.run(["python", "data_training.py"])
    print("Model training complete.")

def run_flask_app():
    """Run the Flask app for inference."""
    print("Starting Flask app...")
    subprocess.run(["python", "app.py"])

if __name__ == "__main__":
    # Step 1: Collect data
    run_data_collection()

    # Step 2: Train the model
    run_data_training()

    # Step 3: Start the Flask app for inference
    run_flask_app()
