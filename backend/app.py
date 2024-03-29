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

@main_bp.route('/run_initial', methods=['POST'])
def run_initial():
    # Extracting data from the request
    data = request.get_json()
    algo = data.get('algo')
    dim = data.get('dim')
    q_inidata = data.get('q_inidata')
    
    # Running the initial iteration function
    result = run_one_iteration_initial(algo, dim, q_inidata)
    
    # Returning the result as JSON
    return jsonify(result), 200

@main_bp.route('/run_normal', methods=['GET'])
def run_normal():
    # Running the normal iteration function
    result = run_one_iteration_normal()
    
    # Returning the result as JSON
    return jsonify(result), 200

# Register blueprints
app.register_blueprint(main_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)
