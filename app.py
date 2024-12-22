from flask import Flask, request, jsonify, send_file
import subprocess
import os

app = Flask(__name__)
@app.route('/run-model', methods=['POST'])
def run_model():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})
        
        file = request.files['image']
        input_path = 'input.png'
        file.save(input_path)

        result = subprocess.run(
            ['python', 'image_multipose_estimation.py'],
            capture_output=True, text=True
        )

        output_path = 'doc/img/output.jpg'
        if os.path.exists(output_path):
            return send_file(output_path, mimetype='image/jpeg')
        else:
            return jsonify({'success': False, 'error': result.stderr})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
