import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pymongo import MongoClient
import pandas as pd

# from .services import get_database, save_csv_to_db, run_optimization, update_model
# from .models import OptimizationInitResponse
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# | CONFIGURING | ----------------------------------------

# loading env vars
load_dotenv()

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

# | UTILITY FUNCTIONS/SERVICES | ----------------------------------


def get_database():
    # MongoDB Atlas connection string
    uri = os.getenv("MONGODB_URI", "your_mongodb_connection_string_here")
    client = MongoClient(uri)  # server_api=ServerApi("1"))
    try:
        client.admin.command("ping")
        logging.info("MongoDB Atlas deployment pinged. Connection successful!")
    except Exception as e:
        logging.error(
            f"Error connecting to MongoDB Atlas! Please log into Atlas console and check settings. Error: {e}"
        )
    return client


def save_csv_to_db(collection, df: pd.DataFrame) -> str:
    # Convert the DataFrame to a dictionary with 'index' orientation
    data_dict = df.to_dict("records")
    # Insert the dictionary as a single document
    result = collection.insert_one({"data": data_dict})
    # Return the ID of the dataset
    logging.info(f"\n New csv saved as document id: {result.inserted_id}")
    return str(result.inserted_id)


# | BACKEND API ENDPOINTS |--------------------------------------------------------------

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://bope-gpt.vercel.app/",
]

# this is currently the main API for BOPE-GPT (under dev), to be hosted and its endpoints hit
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    allow_credentials=True,  # Allows cookies to be sent with requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def main():
    logging.info("BOPE-GPT site loaded")
    return {"message": "CORS is now enabled!"}


# Connect to MongoDB when the app starts
@app.on_event("startup")
async def startup_db_client():
    logging.info("Startup event triggered. Connecting to db...")
    app.mongodb_client = get_database()
    app.mongodb = app.mongodb_client["bope_db"]


# Disconnect from MongoDB when the app shuts down
@app.on_event("shutdown")
async def shutdown_db_client():
    logging.info("Shutdown event triggered.")
    try:
        app.mongodb_client.close()
        logging.info("Connection closed")
    except Exception as e:
        logging.error(f"Error closing db connection: {e}")


@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    logging.info(f"Received file with content type: {file.content_type}")

    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        df = pd.read_csv(file.file)
        logging.info(f"CSV file uploaded by user. CSV->dataframe = {df!r}")
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    try:
        # Save the CSV data into MongoDB
        collection = app.mongodb["datasets"]
        dataset_id = save_csv_to_db(collection, df)
        logging.info(f"Dataset saved to MongoDB with ID: {dataset_id}")
        return {"message": "Dataset uploaded successfully", "dataset_id": dataset_id}
    except Exception as e:
        logging.error(f"Error saving CSV to MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Error saving dataset")


# | WIP Endpoints |--------------------------------------------------------------------------------------


"""
@app.post("/initialize-optimization/", response_model=OptimizationInitResponse)
async def initialize_optimization(
    prompt: str = Form(...),
    num_inputs: int = Form(...),
    num_initial_samples: int = Form(...),
    num_samples_per_batch: int = Form(...),
):
    # Initialize the optimization with the given dataset ID
    result = run_optimization(
        db=app.mongodb,
        dataset_id=dataset_id,  # figure out a way to get dataset id from client- via cookies or authentication?
        num_inputs=num_inputs,
        num_initial_samples=num_initial_samples,
        num_samples_per_batch=num_samples_per_batch,
    )
    return result


@app.post("/next-iteration/")
async def next_iteration():
    # Execute the next iteration of the Bayesian optimization process
    result = update_model()
    return result


@app.get("/results/", response_model=List[OptimizationResult])
async def get_results():
    # Return the current state of the optimization results
    # This would be fetched from wherever you're storing these results (e.g., database)
    # TODO: Fetch these from MongoDB database
    return JSONResponse(content={"results": "Mocked results"})
"""
