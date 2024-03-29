from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS
from config import Config

from models.model_run import run_one_iteration_initial, run_one_iteration_normal

app = Flask(__name__)
app.config.from_object(Config)

# Initialize CORS with your app
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"])

# Blueprint setup
main_bp = Blueprint('main', __name__)

# Initializing global variables
best_vals = {}
data = {}

@main_bp.route('/run_initial', methods=['POST'])
def run_initial():
    # Extracting data from the request
    data = request.get_json()
    algo = data.get('algo')
    dim = data.get('dim')
    q_inidata = data.get('q_inidata')
    prompt = data.get('prompt', '')  # Extracting prompt value from the request
    # Running the initial iteration function
    vals_tmp, data_tmp = run_one_iteration_initial(algo, dim, q_inidata,prompt=prompt)
    
    # Here, update `best_vals` and `data` based on `result` as needed
    # For example:
    # best_vals['initial'] = result.get('best_value')
    # data['initial'] = result
    
    # Returning the result as JSON
    return jsonify(result), 200

@main_bp.route('/run_normal', methods=['GET'])
def run_normal():
    # Extract a query parameter named 'prompt' if needed
    data = request.get_json()
    algo = data.get('algo')
    dim = data.get('dim')
    q_inidata = data.get('q_inidata')
    prompt = data.get('prompt', '')  #
    # Running the normal iteration function
    vals_tmp, data_tmp= run_one_iteration_normal(algo, dim, q_inidata, best_vals,data,prompt=prompt)
    
    # Here, update `best_vals` and `data` based on `result` as needed
    # For example:
    # best_vals['normal'] = result.get('best_value')
    # data['normal'] = result
    
    # Returning the result as JSON
    return jsonify(result), 200

# Register blueprints
app.register_blueprint(main_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
