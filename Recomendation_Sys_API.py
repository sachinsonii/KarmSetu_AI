import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import uvicorn
import os
from pydantic import BaseModel
import pymongo
def load():
    job_postings_df=pd.read_csv('projects_data.csv')
    candidate_profiles_df=pd.read_csv('users_data.csv')

    # Encode the skills using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Directly work with the columns without using eval
    job_postings_df['technologies'] = job_postings_df['technologies'].apply(lambda x: x if isinstance(x, list) else eval(x))
    candidate_profiles_df['skill'] = candidate_profiles_df['skill'].apply(lambda x: x if isinstance(x, list) else eval(x))


    job_skills_encoded = mlb.fit_transform(job_postings_df['technologies'])
    candidate_skills_encoded = mlb.transform(candidate_profiles_df['skill'])

    # Convert the encoded skills into DataFrames
    job_skills_df = pd.DataFrame(job_skills_encoded, columns=mlb.classes_)
    candidate_skills_df = pd.DataFrame(candidate_skills_encoded, columns=mlb.classes_)

    # Combine the encoded skills with the original DataFrame
    job_postings_encoded_df = pd.concat([job_postings_df, job_skills_df], axis=1)
    candidate_profiles_encoded_df = pd.concat([candidate_profiles_df, candidate_skills_df], axis=1)

    # Drop the original skills columns
    job_postings_encoded_df = job_postings_encoded_df.drop(['technologies'], axis=1)
    candidate_profiles_encoded_df = candidate_profiles_encoded_df.drop(['skill'], axis=1)
    return job_postings_encoded_df, candidate_profiles_encoded_df, mlb


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

# Function to match a specific freelancer to all jobs
def match_freelancer_to_jobs(profile_id, top_n=5):
    try:
        job_postings_encoded_df, candidate_profiles_encoded_df, mlb=load()
        # Find the profile with the given profile_id
        candidate = candidate_profiles_encoded_df[candidate_profiles_encoded_df['_id'] == profile_id]
        if candidate.empty:
            raise IndexError("Profile ID not found")
        candidate = candidate.iloc[0]
        candidate_skills = candidate[mlb.classes_].values

        # Calculate match scores for all jobs
        match_scores = []
        for i, job in job_postings_encoded_df.iterrows():
            job_skills = job[mlb.classes_].values
            score = np.dot(candidate_skills, job_skills)
            match_scores.append({
                'Job ID': job['_id'],
                'Title': job['projectCategory'],
                'Match Score': score
            })

        # Convert match scores to DataFrame and sort by score
        match_scores_df = pd.DataFrame(match_scores).sort_values(by='Match Score', ascending=False).head(top_n)

        # Extract the job IDs from the top recommendations
        job_ids = match_scores_df['Job ID'].tolist()
        return job_ids
    except IndexError:
        return []  # Return an empty list if the profile_id is not found

# Function to match a specific job to all freelancers
def match_job_to_freelancers(job_id, top_n=5):
    try:
        job_postings_encoded_df, candidate_profiles_encoded_df, mlb=load()
        # Find the job with the given job_id
        job = job_postings_encoded_df[job_postings_encoded_df['_id'] == job_id]
        if job.empty:
            raise IndexError("Job ID not found")
        job = job.iloc[0]
        job_skills = job[mlb.classes_].values

        # Calculate match scores for all freelancers
        match_scores = []
        for i, candidate in candidate_profiles_encoded_df.iterrows():
            candidate_skills = candidate[mlb.classes_].values
            score = np.dot(job_skills, candidate_skills)
            match_scores.append({
                'Profile ID': candidate['_id'],
                'Name': candidate['fullname'],
                'Match Score': score
            })

        # Convert match scores to DataFrame and sort by score
        match_scores_df = pd.DataFrame(match_scores).sort_values(by='Match Score', ascending=False).head(top_n)

        # Extract the freelancer IDs from the top recommendations
        freelancer_ids = match_scores_df['Profile ID'].tolist()
        return freelancer_ids
    except IndexError:
        return []  # Return an empty list if the job_id is not found

@app.get("/freelancer/{profile_id}")
def get_job_recommendations(profile_id: str):
    job_ids = match_freelancer_to_jobs(profile_id)
    if not job_ids:
        raise HTTPException(status_code=404, detail="Profile not found or no matches found")
    return {"_id": job_ids}

@app.get("/job/{job_id}")
def get_freelancer_recommendations(job_id: str):
    freelancer_ids = match_job_to_freelancers(job_id)
    if not freelancer_ids:
        raise HTTPException(status_code=404, detail="Job not found or no matches found")
    return {"_id": freelancer_ids}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use the port provided by Render
    uvicorn.run(app, host="0.0.0.0", port=port)

