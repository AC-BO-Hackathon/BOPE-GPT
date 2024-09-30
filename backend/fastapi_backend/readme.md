# README for FastAPI backend for BOPE-GPT

Some stuff to help understand this. This is the FastAPI-based backend for the BOPE-GPT App -The actual code that lets you run the optimization process on a dataset of your choosing and interact with Cohere for the preference selection via LLM. Running this FastAPI app exposes CORS-enabled API endpoints to easily run the BOPE process on a pre-processed dataset. 

Use this however you want- either use the Next.js based frontend (go to the live Vercel link) if you want a clean UI interface and automatic data visualization, or alternatively directly hit the API endpoints on the backend (here) through Postman or locally. 

## Structure 

- The main FastAPI app file is `main.py` 
- `run_bope.py` has the entire EUBO-LLM algo set up and it runs- but only on the pretrained Fischer ground truth ANN model (weights for this are stored in `fischer_ann_weights.json`) and with a preset prompt. So this currently only optimises the Fischer Tropsch dataset. Support for other datasets has to be added and this integrated into the current FastAPI app in `main.py`. A log of a sample run of this script is in `run_bope_log.txt`
- Update: `bope_functions.py` is the adapted version of `run_bope.py` with functions callable by the API endpoint functions in `main.py`
- `helpers.py` has serialization/deserialization helper functions. Is important because MongoDB doc storage and returning API responses to the frontend require more primitive datatypes but `bope_functions.py` functions require tensor data types. 
- `models.py` holds all the Pydantic schema for this FastAPI app

## Key Dependencies

- A full list of dependencies can be viewed in the `requirements.txt` file 
- 'FastAPI' for the backend API endpoints
- 'Motor' as an async MongoDB driver for database interactions 
- 'Pydantic' for data scheme definition
- 'Cohere' for the LLM preference selection interactions 

## How to Run Locally 

- Rename `.mockenv` to `.env` and enter a MongoDB Atlas URI and Cohere API Key
- Install dependencies with `pip install -r requirements.txt` (Make sure you're on the `..fastapi_backend/src` folder)
- Start the FastAPI server with `fastapi dev main.py`  (Development mode) 
- Or start a production grade Uvicorn server with `python main.py` 

## Testing 

You can test out the API once you've got the server up and running by going to the API docs FastAPI auto-creates and trying out the endpoints there with these examples of request bodies:

- `upload_dataset` endpoint:
Download the `Fischer-processed-csv` from `insert-github-link-here` and upload here. Copy the `state_id` to test the other endpoints. 

- `initialize_bope` endpoint: (Update: Default values + Prompter text has been added to this endpoint, visible in the docs)
```
{
  "llm_prompt": "Insert-Prompt-Here",
  "num_inputs": 4,
  "num_initial_samples": 5,
  "num_initial_comparisons": 10,
  "state_id": "" # Put the state_id received from the `upload_dataset` endpoint first 
}
```

- For other endpoints: Check out the FastAPI auto-generated docs + make sure to use the `state_id` gained after hitting `upload_dataset` 

## More Info 

Look up the official FastAPI docs if you didn't get something here:

- FastAPI Docs: https://fastapi.tiangolo.com/ 

## Miscellaneous Notes for Further Development/Refinement: 

- All main FastAPI functions (`main.py`) are *async* but some functions in `bope_functions.py` are *sync* due to computation-heavy requirements. Just something to keep in mind while adding/modifying functionality- since only async functions can await other async functions, caller functions with at least one async callee function should be *async*.

## To Do:

- Shift to a more modern Python dependency manager like Poetry probably (and a `.toml` file) 