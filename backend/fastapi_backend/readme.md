# README for FastAPI backend for BOPE-GPT

Some details to help understand the structure of the backend. This is the FastAPI-based backend for the BOPE-GPT App -The actual code that lets you run the optimization process on a dataset of your choosing and interact with Cohere for the preference selection via LLM. Running this FastAPI app exposes CORS-enabled API endpoints to easily run the BOPE process on a pre-processed dataset. 

Use this however you want- either use the Next.js based frontend (go to the live Vercel link) if you want a clean UI interface and automatic data visualization, or alternatively directly hit the API endpoints on the backend through Postman or locally if you run this your computer. To do so through Postman, make sure to add to the CORS origins your own IP address or `[*]` (representing any IP) in `src/main.py`. 

## Structure 

- The main FastAPI app file is `main.py` 
- `run_bope.py` has the entire EUBO-LLM algo set up and it runs- but only on the pretrained Fischer ground truth ANN model (weights for this are stored in `fischer_ann_weights.json`) and with a preset prompt. So this currently only optimises the Fischer Tropsch dataset. Support for other datasets has to be added and this integrated into the current FastAPI app in `main.py`. A log of a sample run of this script is in `run_bope_log.txt`
- `bope_functions.py` is the adapted version of `run_bope.py` with functions callable by the API endpoint functions in `main.py`. This has the bulk of the actual computing and LLM-calling functions, and also generates visualizations. 
- `helpers.py` has serialization/deserialization helper functions, Matplotlib local visualization generation functions and a log summarizer function. Serializing/deserializing is important because MongoDB doc storage and returning API responses to the frontend require more primitive datatypes but `bope_functions.py` functions require more complex ones for efficiency- tensor data types, which have to be converted before storage in MongoDB. 
- `models.py` holds all the Pydantic schema for this FastAPI app
- `pyproject.toml` holds dependencies and these need to be installed before usage. 

## Key Dependencies

- ~~~A full list of dependencies can be viewed in the `requirements.txt` file.~~~ EDIT: Poetry is now being used as a dependency manager. Check `pyproject.toml` and `poetry.lock` for dependencies. 
- 'FastAPI' for the backend API endpoints
- 'Motor' as an async MongoDB driver for database interactions 
- 'Pydantic' for data scheme definition
- 'Cohere' for the LLM preference selection interactions 
- 'MongoDB' for storage of the input dataset, current BOPE state and visualization data for the same. 

## Development Dependencies

- 'Black' for code formatting
- 'Pylint' for code linting

## How to Run Locally 

- Rename `.mockenv` to `.env` and enter a MongoDB Atlas URI and Cohere API Key
- ~~~Install dependencies with `pip install -r requirements.txt` (Make sure you're on the `..fastapi_backend/src` folder)~~ Install dependencies with `poetry install`. 
- Start the FastAPI server with `fastapi dev main.py`  (Development mode) 
- Or start a production grade Uvicorn server with `python main.py` or `fastapi main.py`. For better performance with multiple users, Nginx + a Supervisor script starting a Gunicorn process with Uvicorn workers approach is taken in the live deployed instance of the BOPE-GPT backend though and this can be done locally as well. 

## Testing 

You can test out the API once you've got the server up and running by going to the API docs FastAPI auto-creates (you'll see the link if you open `fastapi dev main.py` and trying out the endpoints there with these examples of request bodies:

- `upload_dataset` endpoint:
Download the `Fischer-processed-csv` from `https://github.com/AC-BO-Hackathon/BOPE-GPT/blob/main/data/fischer_data_processed.csv` and upload here. Copy the `state_id` to test the other endpoints. 

- `initialize_bope` endpoint: (Update: Default values + Prompter text has been added to this endpoint, visible in the docs)
```
{
  "llm_prompt": "Insert-Prompt-Here",
  "num_inputs": 4,
  "num_initial_samples": 5,
  "num_initial_comparisons": 10,
  "enable_flexible_prompt": False,
  "state_id": "" # Put the state_id received from the `upload_dataset` endpoint first 
}
```

- For other endpoints: Check out the FastAPI auto-generated docs + make sure to use the `state_id` gained after hitting `upload_dataset` 

- As stated previously, Postman can be used as well for testing the API, but make sure to edit allowed CORS domains in `src/main.py` to allow requests from anywhere. 

## More Info 

Look up the official FastAPI docs if you didn't get something here:

- FastAPI Docs: https://fastapi.tiangolo.com/ 

Or more about the BOPE process: 

- https://botorch.org/tutorials/bope 

## Miscellaneous Notes for Further Development/Refinement: 

- All main FastAPI functions (`main.py`) are *async* but some functions in `bope_functions.py` are *sync* due to computation-heavy requirements. Just something to keep in mind while adding/modifying functionality- since only async functions can await other async functions, caller functions with at least one async callee function should be *async*.

## To Do:

- Add a "generate visualizations" button for explicit mention of when to generate current Pairwise GP visualizations- as this is computationally very expensive/blocking and the app right now generates it after every iteration. 

- Authentication system + persistent BOPE-states after closing and reopening a browser instance of BOPE-GPT. Currently a state-id is used that is uniquely created every time a new dataset is uploaded on a browser instance. 