from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymongo
import pandas as pd
import uvicorn
from pyngrok import ngrok
import threading

app = FastAPI()

# MongoDB connection URL
MONGO_URI = "mongodb+srv://surajsingh2004ok:76U-iMssyKKT2Vp@cluster0.1vhho4v.mongodb.net/karmsetu?"
DATABASE_NAME = "karmsetu"
USERS_COLLECTION = "users"
PROJECTS_COLLECTION = "projects"

class RunRequest(BaseModel):
    run_code: bool

@app.post("/fetch-data/")
async def fetch_data(request: RunRequest):
    if request.run_code:
        try:
            # Connect to MongoDB Atlas
            client = pymongo.MongoClient(MONGO_URI)
            db = client[DATABASE_NAME]

            # Fetch users data
            users_data = list(db[USERS_COLLECTION].find())
            # Fetch projects data
            projects_data = list(db[PROJECTS_COLLECTION].find())

            # Convert to DataFrames
            users_df = pd.DataFrame(users_data)
            projects_df = pd.DataFrame(projects_data)

            # Save users_data to CSV
            users_df.to_csv('users_data.csv', index=False)

            # Save projects_data to CSV
            projects_df.to_csv('projects_data.csv', index=False)

            return {"message": "Data fetched and saved to CSV files successfully."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"message": "Code execution skipped as per request."}
