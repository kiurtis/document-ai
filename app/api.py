from flask import Flask, request, jsonify
from app.authentication import auth
from werkzeug.utils import secure_filename
import os
from loguru import logger
from ai_documents.analysis.entities import ArvalClassicGPTDocumentAnalyzer, ArvalClassicGeminiDocumentAnalyzer
from ai_documents.validation.entities import ResultValidator

app = Flask(__name__)

# Configuration for file upload
UPLOAD_FOLDER = 'data/tmp'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
@auth.login_required
def get_status():
    # n_thread = threading.active_count()
    # return {"message":f"The API is up and running. Number of threads: {n_thread}!"}
    return {"message": f"The API is up and running."}
@app.route('/validate_document', methods=['POST'])
@auth.login_required
def validate_document():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(request)
        print(request.form)
        # Extract additional parameters from the form
        valet_name = request.form.get('valet_name', None)
        plate_number = request.form.get('plate_number', None)
        from_concessionaire = request.form.get('from_concessionaire', None)
        to_concessionaire = request.form.get('to_concessionaire', None)
        model_type = request.form.get('model', 'GPT')  # Default to 'GPT' if not specified

        try:
            # Select the document analyzer based on the MODEL type
            if model_type.upper() == 'GPT':
                document_analyzer = ArvalClassicGPTDocumentAnalyzer(filename, file_path)
            elif model_type.upper() == 'GEMINI':
                document_analyzer = ArvalClassicGeminiDocumentAnalyzer(filename, file_path)
            else:
                return jsonify({'error': f"Unsupported model type {model_type}"}), 400

            document_analyzer.analyze()

            # Assuming ResultValidator can be used for both models
            result_validator = ResultValidator(document_analyzer.results, plate_number=plate_number,
                                               valet_name=valet_name, from_concessionaire=from_concessionaire,
                                               to_concessionaire=to_concessionaire)
            result_validator.validate()

            dict_to_save = document_analyzer.results.copy()
            dict_to_save['refused_causes'] = result_validator.refused_causes
            dict_to_save['Validated'] = result_validator.validated

            return jsonify(dict_to_save), 200

        except Exception as e:
            logger.error(f"Error {e} while analyzing {filename}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
