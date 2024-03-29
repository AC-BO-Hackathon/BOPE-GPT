# main.py

from flask import Blueprint, request, jsonify
from models.bope import BOPEModel
from queue import Queue

main_bp = Blueprint('main', __name__)
model_instances = {}
model_queue = Queue()

@main_bp.route('/initialize', methods=['POST'])
def initialize_bope():
    data = request.get_json()
    dataset = data['dataset']
    num_inputs = data['num_inputs']
    num_init_samples = data['num_init_samples']
    llm_prompt = data['llm_prompt']

    # Check if maximum models limit is reached
    if len(model_instances) >= app.config['MAX_MODELS']:
        return jsonify({'message': 'High app demand, please try after some time'}), 503

    model = BOPEModel(dataset, num_inputs, num_init_samples, llm_prompt)
    model_id = len(model_instances)
    model_instances[model_id] = model

    return jsonify({'model_id': model_id, 'visualization_data': model.get_visualization_data()})

@main_bp.route('/iterate', methods=['POST'])
def iterate_bope():
    data = request.get_json()
    model_id = data['model_id']

    if model_id not in model_instances:
        return jsonify({'error': 'Invalid model ID'}), 400

    model = model_instances[model_id]
    preference_feedback = data['preference_feedback']
    model.update_models(preference_feedback)
    model.optimize_acquisition_funcs()
    visualization_data = model.get_visualization_data()

    return jsonify({'visualization_data': visualization_data})

@main_bp.route('/visualization', methods=['GET'])
def get_visualization_data():
    model_id = request.args.get('model_id')

    if model_id not in model_instances:
        return jsonify({'error': 'Invalid model ID'}), 400

    model = model_instances[model_id]
    visualization_data = model.get_visualization_data()

    return jsonify({'visualization_data': visualization_data})