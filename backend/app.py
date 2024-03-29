# app.py 

from flask import Flask
from flask_cors import CORS  
from config import Config
from routes.main import main_bp

app = Flask(__name__)
app.config.from_object(Config)

# Initialize CORS with your app
CORS(app, origins="*", methods=["GET", "POST"])

# Register blueprints
app.register_blueprint(main_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(debug=True)

