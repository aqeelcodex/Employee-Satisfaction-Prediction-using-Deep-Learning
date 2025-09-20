from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd

# Initialize FastAPI app with metadata
app = FastAPI(
    title= "Deploying Employee Satisfaction Prediction Model with FastAPI",
    description= "This project demonstrates how to deploy a deep learning model for predicting employee satisfaction using FastAPI. The workflow includes loading trained models, preprocessing pipelines (scaler and encoders), and serving predictions through a lightweight, high-performance API",
    version= "1.0.0"
)

# Load trained model and preprocessing objects
try:
    with open("models.pkl", "rb") as file:
        model_loaded = pickle.load(file)
    with open("scale.pkl", "rb") as file:
        scale_loaded = pickle.load(file)
    with open("ohe.pkl", "rb") as file:
        ohe_loaded = pickle.load(file)
    with open("oe.pkl", "rb") as file:
        oe_loaded = pickle.load(file)
except FileNotFoundError as e:
    print(f"Some file is not Found: {e}")

# Define input schema using Pydantic for request validation
class EmployeInputs(BaseModel):
    Department: str
    Gender: str
    Age: int
    Job_Title: str
    Years_At_Company: int
    Education_Level: str
    Performance_Score: int
    Monthly_Salary: float
    Work_Hours_Per_Week: int
    Projects_Handled: int
    Overtime_Hours: int
    Sick_Days: int
    Remote_Work_Frequency: int
    Team_Size: int
    Training_Hours: int
    Promotions: int
    Resigned: int  # encoded as int (0/1)

# Define response schema for consistent API output
class PredictionResponse(BaseModel):
    predicted_value: float
    employe_features: dict

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Employee Performance Prediction Model! Go to /docs for Swagger UI."}

# Prediction endpoint
@app.post("/Prediction", response_model=PredictionResponse)
def prediction_func(employe: EmployeInputs):
    try: 
        # Convert input JSON into DataFrame
        user_input = pd.DataFrame([employe.dict()])
        
        # Apply Ordinal Encoding on Education_Level
        user_input[["Education_Level"]] = oe_loaded.transform(user_input[["Education_Level"]])

        # Apply OneHotEncoding on categorical columns
        ohe_cols = ["Department", "Gender", "Job_Title"]
        ohe_features = ohe_loaded.transform(user_input[ohe_cols])
        ohe_feature_names = ohe_loaded.get_feature_names_out(ohe_cols)
        df_ohe_features = pd.DataFrame(ohe_features, columns= ohe_feature_names, index= user_input.index)
        
        # Merge encoded features with the rest of the data
        user_input = pd.concat([user_input.drop(columns= ohe_cols), df_ohe_features], axis=1)

        # Scale numerical features
        input_scaled = scale_loaded.transform(user_input)       

        # Make prediction using loaded model
        prediction = model_loaded.predict(input_scaled)[0][0]

        # Return structured response
        return PredictionResponse(
            predicted_value= prediction,
            employe_features= employe.dict()
        )
    except Exception as e:
       # If something fails, return 500 error
       raise HTTPException(status_code=500, detail= f"Something is going wrong. {str(e)}")
    
# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host= "127.0.0.1", port= 8000)
