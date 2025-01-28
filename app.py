from flask import Flask, request, send_from_directory, jsonify, session, redirect, url_for
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import whisper
import torch
import threading
from pathlib import Path
import warnings
import time
import logging
import sys
from anthropic import Anthropic
from dotenv import load_dotenv
from functools import wraps
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables from .env file
load_dotenv()

# Add Homebrew bin directory to PATH
os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ.get('PATH', '')

# Filter out the specific FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

app = Flask(__name__, static_url_path='')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRANSCRIPT_FOLDER'] = 'transcripts'
app.secret_key = os.urandom(24)  # For session management

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
)

# Login credentials from environment variables
VALID_USERNAME = os.getenv('ADMIN_USERNAME')
VALID_PASSWORD = os.getenv('ADMIN_PASSWORD')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('serve_login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per hour")  # 5 attempts per hour
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Invalid username or password'}), 401
    
    return redirect(url_for('serve_login'))

@app.route('/login.html')
def serve_login():
    if 'logged_in' in session:
        return redirect(url_for('serve_index'))
    return send_from_directory('static', 'login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('serve_login'))

@app.route('/')
@login_required
def serve_index():
    return send_from_directory('static', 'index.html')

@app.route('/transcript.html')
@login_required
def serve_transcript_page():
    return send_from_directory('static', 'transcript.html')

# Create uploads directory if it doesn't exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['TRANSCRIPT_FOLDER']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

class TranscriptionThread(threading.Thread):
    def __init__(self, audio_path, transcript_path):
        threading.Thread.__init__(self)
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        
    def run(self):
        try:
            # Load the model
            model = whisper.load_model("base")
            
            # Transcribe
            result = model.transcribe(self.audio_path)
            
            # Save transcript
            with open(self.transcript_path, "w") as f:
                f.write(result["text"])
                
        except Exception as e:
            # If there's an error, write it to the transcript file
            with open(self.transcript_path, "w") as f:
                f.write(f"Error during transcription: {str(e)}")

def transcribe_audio(audio_path, timestamp):
    try:
        # Load the model
        model = whisper.load_model("base")
        
        # Transcribe
        result = model.transcribe(audio_path)
        
        # Save transcript
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f'transcript_{timestamp}.txt')
        with open(transcript_path, "w") as f:
            f.write(result["text"])
            
    except Exception as e:
        # If there's an error, write it to the transcript file
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f'transcript_{timestamp}.txt')
        with open(transcript_path, "w") as f:
            f.write(f"Error during transcription: {str(e)}")

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    title = request.form.get('title', 'Untitled Recording')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'recording_{timestamp}.wav'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Save the title to a metadata file
        metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.meta')
        with open(metadata_path, 'w') as f:
            json.dump({'title': title}, f)
        
        # Start transcription in the background
        transcription_thread = TranscriptionThread(filepath, os.path.join(app.config['TRANSCRIPT_FOLDER'], f'transcript_{timestamp}.txt'))
        transcription_thread.start()
        
        return jsonify({'message': 'File uploaded successfully'}), 200
    
    return jsonify({'error': 'Error uploading file'}), 400

@app.route('/recordings')
@login_required
def list_recordings():
    recordings = []
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.wav'):
            timestamp = filename.replace('recording_', '').replace('.wav', '')
            
            # Get the title from metadata file
            title = 'Untitled Recording'
            metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.meta')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        title = metadata.get('title', 'Untitled Recording')
                except:
                    pass
            
            # Check for transcript
            transcript_exists = os.path.exists(os.path.join(app.config['TRANSCRIPT_FOLDER'], f'transcript_{timestamp}.txt'))
            
            recordings.append({
                'filename': filename,
                'url': f'/uploads/{filename}',
                'title': title,
                'timestamp': timestamp,
                'has_transcript': transcript_exists
            })
    
    return jsonify(sorted(recordings, key=lambda x: x['timestamp'], reverse=True))

@app.route('/view_transcript/<timestamp>')
@login_required
def view_transcript(timestamp):
    try:
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f'transcript_{timestamp}.txt')
        print(f"Viewing transcript: {transcript_path}")  # Debug log
        
        if not os.path.exists(transcript_path):
            print(f"Transcript not found: {transcript_path}")  # Debug log
            return "Transcript not ready yet. Please wait a few seconds and try again.", 404
            
        with open(transcript_path, 'r') as f:
            transcript_text = f.read()
            print(f"Transcript content length: {len(transcript_text)}")  # Debug log
        
        if not transcript_text.strip():
            return "Transcript is empty. There might have been an error in processing.", 500
            
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transcript</title>
            <style>
                body {{ font-family: -apple-system; max-width: 800px; margin: 20px auto; padding: 20px; }}
                .container {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .transcript {{ white-space: pre-wrap; line-height: 1.5; }}
                .back {{ display: inline-block; padding: 10px 20px; background: #007AFF; color: white; 
                        text-decoration: none; border-radius: 20px; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="transcript">{transcript_text}</div>
                <a href="/" class="back">Back</a>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        return f"Error viewing transcript: {str(e)}", 500

@app.route('/transcript/<timestamp>')
@login_required
def get_transcript(timestamp):
    transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f'transcript_{timestamp}.txt')
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r') as f:
            return f.read()
    return 'Transcript not found', 404

@app.route('/recording_title/<timestamp>')
@login_required
def get_recording_title(timestamp):
    try:
        metadata_path = os.path.join(app.config['UPLOAD_FOLDER'], f'recording_{timestamp}.wav.meta')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return jsonify({'title': metadata.get('title', 'Untitled Recording')})
        return jsonify({'title': 'Untitled Recording'})
    except Exception as e:
        return jsonify({'title': 'Untitled Recording'})

@app.route('/delete/<timestamp>', methods=['DELETE'])
@login_required
def delete_recording(timestamp):
    try:
        # Delete recording file
        recording_path = os.path.join(app.config['UPLOAD_FOLDER'], f'recording_{timestamp}.wav')
        if os.path.exists(recording_path):
            os.remove(recording_path)
            
        # Delete metadata file if it exists
        metadata_path = recording_path + '.meta'
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        # Delete transcript file if it exists
        transcript_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f'transcript_{timestamp}.txt')
        if os.path.exists(transcript_path):
            os.remove(transcript_path)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
@login_required
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/transcripts/<path:filename>')
@login_required
def serve_transcript(filename):
    return send_from_directory(app.config['TRANSCRIPT_FOLDER'], filename)

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data or 'context' not in data:
            return jsonify({'error': 'Invalid request data'}), 400

        message = data['message']
        context = data['context']

        # Initialize Anthropic client
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Use Claude to generate a response
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"Here is the transcript:\n\n{context}\n\nUser question: {message}"
                }
            ]
        )

        return jsonify({
            'response': response.content[0].text
        })
    except Exception as e:
        print(f"Chat error: {str(e)}", file=sys.stderr)  # Log the error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, host='127.0.0.1')
