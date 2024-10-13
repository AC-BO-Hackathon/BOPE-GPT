import logging
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
import pandas as pd
from typing import List, Optional, Any
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import torch
import json
from pydantic import BaseModel
from bson import ObjectId
import time

# Import Pydantic models
from models import (
    InitializeBOPERequest,
    RunNextIterationRequest,
    BopeState,
    State,
    SerializedBopeState,
    SerializedState,
    UploadedDataset,
)

# Import BOPE-related functions
from bope_functions import initialize_bope, run_next_iteration

# Import helper functions
from helpers import (
    serialize_bope_state,
    deserialize_bope_state,
    serialize_state,
    brief_summary,
)

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


class ExcludeWatchfilesInfoFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.name == "watchfiles.main"
            and "1 change detected" in record.getMessage()
        )


watchfiles_logger = logging.getLogger("watchfiles.main")
watchfiles_logger.addFilter(ExcludeWatchfilesInfoFilter())

# FastAPI app setup
app = FastAPI()

# CORS middleware setup
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:3001",
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
    logging.info(f"Async Motor client initialized: {client}")
    return client.bope_db


# WIP: replace `on_event` (deprecated) with fastapi lifecycle event handler
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
        uploaded_dataset = UploadedDataset(
            data=data_dict,
            column_names=column_names,
            uploaded_at=datetime.now(timezone.utc),
        )
        result = await collection.insert_one(uploaded_dataset.model_dump())
        dataset_id = str(result.inserted_id)
        logging.info(f"New csv saved as document id: {dataset_id}")

        # Create a new state and associate it with the dataset
        state_id = await create_new_state(column_names, bounds)
        state_collection = app.mongodb.bope_states
        await state_collection.update_one(
            {"_id": ObjectId(state_id)}, {"$set": {"dataset_id": dataset_id}}
        )

        logging.info(f"New State created with state id: {state_id}")

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
async def create_new_state(column_names: List[str], bounds: List[List[float]]) -> str:
    collection = app.mongodb.bope_states
    new_state = State(
        created_at=datetime.now(timezone.utc),
        dataset_id=None,
        column_names=column_names,
        bounds=bounds,
        bope_state=None,
    )
    result = await collection.insert_one(new_state.model_dump())
    return str(result.inserted_id)


async def get_state(state_id: str) -> Optional[SerializedState]:
    collection = app.mongodb.bope_states
    state_doc = await collection.find_one({"_id": ObjectId(state_id)})
    if state_doc is None:
        return None
    return SerializedState(**state_doc)


# mongodb handlers for bope_state (subset of overall state)
async def update_bope_state(
    state_id: str, bope_state: BopeState, iteration_duration: float
):  # serializes bope state and updates mongodb state doc
    collection = app.mongodb.bope_states
    bope_state.last_iteration_duration = iteration_duration
    bope_state.updated_at = datetime.now(timezone.utc)
    serialized_state = serialize_bope_state(bope_state).model_dump()
    print(f"\n serialized_state = {serialized_state}")
    await collection.update_one(
        {"_id": ObjectId(state_id)}, {"$set": {"bope_state": serialized_state}}
    )


async def retrieve_bope_state(
    state_id: str,
) -> Optional[BopeState]:  # deserializes bope state retrieved from mongodb
    collection = app.mongodb.bope_states
    state_doc = await collection.find_one({"_id": ObjectId(state_id)})
    if state_doc is None or state_doc.get("bope_state") is None:
        return None

    serialized_bope_state: SerializedBopeState = SerializedBopeState(
        **state_doc["bope_state"]
    )
    bope_state: BopeState = deserialize_bope_state(serialized_bope_state)
    return bope_state


@app.post("/initialize_bope/")
async def initialize_bope_endpoint(request: InitializeBOPERequest):
    try:
        start_time = time.time()
        dim = request.num_inputs
        q_inidata = request.num_initial_samples
        q_comp_ini = request.num_initial_comparisons

        logging.info(
            "Initializing BOPE with llm_prompt='%s', enable_llm_explanations=%s, enable_flexible_prompt=%s, num_inputs=%d, q_inidata=%d, q_comp_ini=%d, state_id=%s",
            request.llm_prompt,
            request.enable_llm_explanations,
            request.enable_flexible_prompt,
            request.num_inputs,
            request.num_initial_samples,
            request.num_initial_comparisons,
            request.state_id,
        )

        state: SerializedState = await get_state(request.state_id)
        state = state.model_dump()
        bounds = state.get("bounds")

        column_names = state.get("column_names")

        bope_state: BopeState = await initialize_bope(
            dim, q_inidata, q_comp_ini, bounds, column_names
        )
        end_time = time.time()
        iteration_duration = round(end_time - start_time, 5)

        # Update the state in MongoDB
        await update_bope_state(request.state_id, bope_state, iteration_duration)

        # Retrieve updated bope state and serialize

        bope_state: BopeState = await retrieve_bope_state(request.state_id)

        response_bope_state: SerializedBopeState = serialize_bope_state(bope_state)

        summarizing_start_time = time.time()
        brief_bope_state = brief_summary(response_bope_state.model_dump())
        summarizing_end_time = time.time()
        print(
            f"\n Loggable summarizing time: {summarizing_end_time - summarizing_start_time}"
        )

        logging.info(
            f"State {request.state_id} \nIteration {bope_state.iteration}\ndetails: {json.dumps(brief_bope_state, indent=2)}"
        )

        return JSONResponse(
            content={
                "message": "BOPE initialized successfully",
                "bope_state": response_bope_state.model_dump_json(),
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
        logging.info(
            "Running next BOPE iteration with llm_prompt='%s', comparison_explanations=%s, enable_flexible_prompt=%s, state_id=%s",
            request.llm_prompt,
            request.comparison_explanations,
            request.enable_flexible_prompt,
            request.state_id,
        )

        start_time = time.time()
        # Retrieve the state from MongoDB
        bope_state: BopeState = await retrieve_bope_state(request.state_id)

        if bope_state is None:
            raise HTTPException(status_code=404, detail="BOPE state not found")

        # Reconstruct the model
        from bope_functions import init_and_fit_model

        _, model = init_and_fit_model(bope_state.X, bope_state.comparisons)

        # Run the next iteration
        bope_state: BopeState = await run_next_iteration(bope_state, model)
        end_time = time.time()
        iteration_duration = round(end_time - start_time, 5)

        # Update the state in MongoDB
        await update_bope_state(request.state_id, bope_state, iteration_duration)

        # Retrieve updated bope state and serialize
        bope_state: BopeState = await retrieve_bope_state(request.state_id)

        response_bope_state: SerializedBopeState = serialize_bope_state(bope_state)

        summarizing_start_time = time.time()
        brief_bope_state = brief_summary(response_bope_state.model_dump())
        summarizing_end_time = time.time()
        print(
            f"\n Loggable summarizing time: {summarizing_end_time - summarizing_start_time}"
        )

        logging.info(
            f"State {request.state_id} \nIteration {bope_state.iteration}\ndetails: {json.dumps(brief_bope_state, indent=2)}"
        )

        return JSONResponse(
            content={
                "message": "Next iteration completed successfully",
                "bope_state": response_bope_state.model_dump_json(),
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
        logging.info(
            "Getting BOPE State with state_id=%s",
            state_id,
        )

        bope_state: BopeState = await retrieve_bope_state(state_id)
        if bope_state is None:
            raise HTTPException(status_code=404, detail="BOPE state not found")

        response_bope_state: SerializedBopeState = serialize_bope_state(bope_state)

        logging.info(
            f"State {state_id}\n Iteration {bope_state.iteration}\ndetails: {json.dumps(response_bope_state.model_dump_json(), indent=2)}"
        )

        return JSONResponse(
            content={
                "bope_state": response_bope_state.model_dump_json(),
            }
        )
    except Exception as e:
        logging.error(f"Error retrieving BOPE state: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving BOPE state: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, reload_excludes=["*.log"])
    logging.info("FastAPI App started in production mode")
