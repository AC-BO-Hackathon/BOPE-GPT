import logging
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
import pandas as pd
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import torch
import json
from pydantic import BaseModel
from bson import ObjectId

# Import BOPE-related functions
from bope_functions import initialize_bope, run_next_iteration

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

# FastAPI app setup
app = FastAPI()

# CORS middleware setup
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "https://bope-gpt.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# MongoDB connection
async def get_database():
    client = AsyncIOMotorClient(
        os.getenv("MONGODB_URI", "your_mongodb_connection_string_here")
    )
    print(f"Async Motor client initialized: {client}")
    return client.bope_db


# Pydantic models for request validation
class InitializeBOPERequest(BaseModel):
    llm_prompt: str = "Enter a prompt here"  # ""
    num_inputs: int = 4  # 4
    num_initial_samples: int = 5  # 5
    num_initial_comparisons: int = 10  # 10
    state_id: str = (
        "Insert whatever state ID received after hitting the `upload_dataset` endpoint"
    )


class RunNextIterationRequest(BaseModel):
    prompt: str
    state_id: str


@app.on_event("startup")
async def startup_db_client():
    app.mongodb = await get_database()


@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongodb.client.close()


@app.get("/")
async def main():
    logging.info("BOPE-GPT site loaded")
    return {"message": "Welcome to BOPE-GPT API!"}


@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        logging.info(f"CSV file uploaded by user. CSV->dataframe shape: {df.shape}")

        # Get column names
        column_names = df.columns.tolist()

        # Calculate input bounds (we'll store all bounds and define the required ones- input_bounds later, in `initialize_bope`)
        bounds = []
        for col in df.columns:
            bounds.append([float(df[col].min()), float(df[col].max())])

    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        raise HTTPException(status_code=400, detail="Invalid CSV file")

    try:
        collection = app.mongodb.datasets
        data_dict = df.to_dict("records")
        result = await collection.insert_one(
            {
                "data": data_dict,
                "column_names": column_names,
                "uploaded_at": datetime.now(timezone.utc),
            }
        )
        dataset_id = str(result.inserted_id)
        logging.info(f"New csv saved as document id: {dataset_id}")

        # Create a new state and associate it with the dataset
        state_id = await create_new_state(column_names, bounds)
        state_collection = app.mongodb.bope_states
        await state_collection.update_one(
            {"_id": ObjectId(state_id)}, {"$set": {"dataset_id": dataset_id}}
        )

        print(f"New State created with state id: {state_id}")

        return {
            "message": "Dataset uploaded successfully",
            "dataset_id": dataset_id,
            "state_id": state_id,
            "column_names": column_names,
        }
    except Exception as e:
        logging.error(f"Error saving CSV to MongoDB: {e}")
        raise HTTPException(status_code=500, detail="Error saving dataset")


# handlers for overall state (inclusive of bope_state if present)
async def create_new_state(column_names, bounds):
    collection = app.mongodb.bope_states
    new_state = {
        "created_at": datetime.now(timezone.utc),
        "dataset_id": None,
        "column_names": column_names,
        "bounds": bounds,  # holds bounds for ALL columns of dataset
        "bope_state": None,
    }
    result = await collection.insert_one(new_state)
    return str(result.inserted_id)


async def get_state(state_id):
    collection = app.mongodb.bope_states
    state_doc = await collection.find_one({"_id": ObjectId(state_id)})
    if state_doc is None:
        return None
    return state_doc


# handlers for bope_state (a subset of overall state)
async def update_bope_state(state_id, bope_state):
    collection = app.mongodb.bope_states
    serialized_state = {
        "X": bope_state["X"].tolist(),
        "comparisons": bope_state["comparisons"].tolist(),
        "best_val": bope_state["best_val"].tolist(),
        "input_bounds": [
            b.tolist() for b in bope_state["input_bounds"]
        ],  # holds bounds for input columns only
        "updated_at": datetime.now(timezone.utc),
    }
    await collection.update_one(
        {"_id": ObjectId(state_id)}, {"$set": {"bope_state": serialized_state}}
    )


async def retrieve_bope_state(state_id):
    collection = app.mongodb.bope_states
    state_doc = await collection.find_one({"_id": ObjectId(state_id)})
    if state_doc is None or state_doc.get("bope_state") is None:
        return None

    bope_state = state_doc["bope_state"]
    state = {
        "X": torch.tensor(bope_state["X"]),
        "comparisons": torch.tensor(bope_state["comparisons"]),
        "best_val": torch.tensor(bope_state["best_val"]),
        "input_bounds": torch.stack(
            [torch.tensor(b) for b in bope_state["input_bounds"]]
        ),  # holds bounds for input columns only
        "model": None,  # We'll need to reconstruct the model
    }
    return state


@app.post("/initialize_bope/")
async def initialize_bope_endpoint(request: InitializeBOPERequest):
    try:
        dim = request.num_inputs
        q_inidata = request.num_initial_samples
        q_comp_ini = request.num_initial_comparisons

        # add a logging.info statement with request parameters:
        logging.info(
            "Initializing BOPE with dim=%d, q_inidata=%d, q_comp_ini=%d, state_id=%s",
            request.num_inputs,
            request.num_initial_samples,
            request.num_initial_comparisons,
            request.state_id,
        )

        state = await get_state(request.state_id)
        bounds = state.get("bounds")
        print(f"\n bounds = {bounds}")

        bope_state = await initialize_bope(dim, q_inidata, q_comp_ini, bounds)

        # Update the state in MongoDB
        await update_bope_state(request.state_id, bope_state)

        # Convert torch tensors to lists for JSON serialization
        response_state = {
            "X": bope_state["X"].tolist(),
            "comparisons": bope_state["comparisons"].tolist(),
            "best_val": bope_state["best_val"].tolist(),
            "input_bounds": [b.tolist() for b in bope_state["input_bounds"]],
        }

        return JSONResponse(
            content={
                "message": "BOPE initialized successfully",
                "bope_state": response_state,
                "state_id": request.state_id,
            }
        )
    except Exception as e:
        logging.error(f"Error initializing BOPE: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing BOPE: {str(e)}"
        )


@app.post("/run_next_iteration/")
async def run_next_iteration_endpoint(request: RunNextIterationRequest):
    try:
        # Retrieve the state from MongoDB
        bope_state = await retrieve_bope_state(request.state_id)
        if bope_state is None:
            raise HTTPException(status_code=404, detail="BOPE state not found")

        # Reconstruct the model
        from bope_functions import init_and_fit_model

        _, bope_state["model"] = init_and_fit_model(
            bope_state["X"], bope_state["comparisons"]
        )

        # Run the next iteration
        bope_state = await run_next_iteration(bope_state)

        # Update the state in MongoDB
        await update_bope_state(request.state_id, bope_state)

        # Convert torch tensors to lists for JSON serialization
        response_state = {
            "X": bope_state["X"].tolist(),
            "comparisons": bope_state["comparisons"].tolist(),
            "best_val": bope_state["best_val"].tolist(),
        }

        return JSONResponse(
            content={
                "message": "Next iteration completed successfully",
                "state": response_state,
                "state_id": request.state_id,
            }
        )
    except Exception as e:
        logging.error(f"Error running next iteration: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error running next iteration: {str(e)}"
        )


@app.get("/get_bope_state/{state_id}")
async def get_bope_state(state_id: str):
    try:
        bope_state = await retrieve_bope_state(state_id)
        if bope_state is None:
            raise HTTPException(status_code=404, detail="BOPE state not found")

        # Convert torch tensors to lists for JSON serialization
        response_state = {
            "X": bope_state["X"].tolist(),
            "comparisons": bope_state["comparisons"].tolist(),
            "best_val": bope_state["best_val"].tolist(),
            "input_bounds": [b.tolist() for b in bope_state["input_bounds"]],
        }

        return JSONResponse(content={"bope_state": response_state})
    except Exception as e:
        logging.error(f"Error retrieving BOPE state: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving BOPE state: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, reload_excludes=["*.log"])
