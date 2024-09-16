# README for FastAPI backend for BOPE-GPT

## Structure 

- For now everything in `main.py` 
- `run_bope.py` has the entire EUBO-LLM algo set up and it runs- but only on the pretrained Fischer ground truth ANN model (weights for this are stored in `fischer_ann_weights.json`) and with a preset prompt. So this only optimises the Fischer Tropsch dataset. Support for other datasets has to be added and this integrated into the current FastAPI app in `main.py`. A log of a sample run of this script is in `run_bope_log.txt`

## Running

- Rename `.mockenv` to `.env` and enter MongoDB Atlas URI 
- Install dependencies with `pip install -r requirements.txt`
- Start FastAPI server with `fastapi dev main.py`  

## More Info 

- FastAPI Docs: https://fastapi.tiangolo.com/ 