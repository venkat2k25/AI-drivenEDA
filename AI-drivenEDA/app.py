from flask import Flask, render_template, request, jsonify
import pandas as pd
from utils.gemini_helper import GeminiAnalyzer
from utils.data_processor import DataProcessor

app = Flask(__name__)


app.config['DATA'] = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_data():
    try:
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            app.config['DATA'] = df
            return jsonify({
                'success': True,
                'message': 'Data uploaded successfully',
                'shape': df.shape,
                'columns': list(df.columns)
            })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if app.config['DATA'] is None:
            return jsonify({'success': False, 'message': 'Please upload data first'})
        
        question = request.json.get('question')
        analyzer = GeminiAnalyzer(app.config['DATA'])
        response = analyzer.analyze_data(question)
        return jsonify({'success': True, 'response': response})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        if app.config['DATA'] is None:
            return jsonify({'success': False, 'message': 'Please upload data first'})
        
        plot_type = request.json.get('plot_type')
        x_column = request.json.get('x_column')
        y_column = request.json.get('y_column')
        title = request.json.get('title', '')
        
        processor = DataProcessor(app.config['DATA'])
        fig = processor.create_visualization(plot_type, x_column, y_column, title)
        return jsonify({
            'success': True,
            'plot': fig.to_json()
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
