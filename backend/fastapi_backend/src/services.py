from .models import OptimizationResult
import random
from .config import settings
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
import pandas as pd
import os

# from dotenv import load_dotenv()

# WIP FILE (SEE README)

# load_dotenv()


def get_database():
    # MongoDB Atlas connection string
    uri = os.getenv("MONGODB_URI", "your_mongodb_connection_string_here")
    # client = MongoClient(uri)
    client = MongoClient(uri, server_api=ServerApi("1"))
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client


def save_csv_to_db(collection, df: pd.DataFrame) -> str:
    # Convert the DataFrame to a dictionary with 'index' orientation
    data_dict = df.to_dict("index")
    # Insert the dictionary as a single document
    result = collection.insert_one({"data": data_dict})
    # Return the ID of the dataset
    return str(result.inserted_id)


# ------------------------------------------------------------------------------------------


def get_llm_client():
    # Initialize the LLM client using the API key from the config
    return None  # Replace with actual client initialization


def run_optimization(prompt, num_inputs, num_initial_samples, num_samples_per_batch):
    # Mocked optimization logic for illustration purposes
    # Here you would initialize the Gaussian process and other necessary models
    result = OptimizationResult(iteration=1, comparisons=[], best_candidate=None)
    return {"status": "initialized", "details": result.dict()}


def update_model():
    # Perform the next iteration of optimization
    # Mocked logic: Randomly select data points, pass to LLM, update model
    chosen = random.randint(1, 10)
    other = random.randint(1, 10)
    result = {
        "status": "iteration completed",
        "results": {"chosen": chosen, "other": other},
    }
    return result
