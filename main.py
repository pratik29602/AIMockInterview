from flask import Flask, render_template, request, Response, jsonify
import cv2
from deepface import DeepFace
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import gtts
import playsound
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from g4f.client import Client
from collections import Counter
import time
import json
from datetime import datetime
app = Flask(__name__)
client = Client()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# TF-IDF Vectorizer for similarity checking
tfidf_vectorizer = TfidfVectorizer()

# Global variables
selected_language = None
topic = None
face_count = 0
emotion_result = None
lock = threading.Lock()  # Lock for thread safety
list_emotions = []
count = 0
confidence_score = 0
questions_attempted = 0
questions_remaining = 10  # Initial number of questions
cheating_detected = False
candidate_name = ''
scheduler = ''
expected_answer = ''
asked_questions = []
questions_answers = {}  # Dictionary to store questions and responses
current_question = '' 
final_emotion = ''
wrong_questions = 0
correct_questions = 0
total_score_percentage = 0.0


def play_voice_response(text):
    """Generate and play a voice response."""
    sound = gtts.gTTS(text, lang='en')
    sound.save("response.mp3")
    playsound.playsound("response.mp3")

# Function to generate a response using g4f (GPT-4)
def generate_response(input_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can adjust the model name here if needed
        messages=[{"role": "user", "content": input_text}],
    )
    return response.choices[0].message.content  # Extract the response text

@app.route('/submit_response', methods=['GET', 'POST'])
def submit_response():
    global topic, count, questions_attempted, questions_remaining, candidate_name, current_question
    global asked_questions
    global questions_answers
    data = request.json
    reply = ''
    bot_response = ''
    # Step 1: Generate Introduction Question
    if count == 0:
        reply = f"Hi {candidate_name}, Please introduce yourself."
        play_voice_response(reply)
        current_question = reply
        count += 1
        return jsonify({'reply': reply})
    
    # Step 2: Generate Topic-Specific Question
    elif count == 1:
        reply = f"As you have chosen {topic}, I will be asking questions on the {topic} programming language."
        play_voice_response(reply)
        bot_response = generate_response(f"Generate a unique one-liner question related to the {topic} programming language. Do not include any introduction, explanation, or extra text. Only generate the question itself and avoid asking questions from this list: {asked_questions}."
)
        while bot_response == 'Request ended with status code 404':
                #  bot_response = generate_response(f"Generate a unique one-liner question related to the {topic} programming language. Ensure that the question is concise, clear, and covers a fundamental concept of the topic. Each question should be short, unique, and directly related to {topic}. Do not include any introduction, explanation, or extra text. Only generate the question itself.")
               bot_response = generate_response(f"Generate a unique one-liner question related to the {topic} programming language. Do not include any introduction, explanation, or extra text. Only generate the question itself and avoid asking questions from this list: {asked_questions}."
)
        
        play_voice_response(bot_response)
        current_question = bot_response
        asked_questions.append(current_question)
        questions_answers[current_question] = ''  # Initialize the dictionary with an empty answer
        count += 1
        return jsonify({'reply': bot_response})

    # Step 3: Handle Candidate's Response and Generate New Questions
    elif count > 1 and count <= 11:
        candidate_answer = data.get('response', '')
        if current_question:
            #questions_answers[current_question] = candidate_answer  # Store the answer for the current question
            asked_questions.append(current_question)
        print(f"Question: {current_question} | Answer: {candidate_answer}")
        questions_answers[current_question] = candidate_answer  # Store the answer for the current question
        bot_response = generate_response(f"Generate a unique one-liner question related to the {topic} programming language. Do not include any introduction, explanation, or extra text. Only generate the question itself and avoid asking questions from this list: {asked_questions}."
)
         
        while bot_response == 'Request ended with status code 404':
                #  bot_response = generate_response(f"Generate a unique one-liner question related to the {topic} programming language. Ensure that the question is concise, clear, and covers a fundamental concept of the topic. Each question should be short, unique, and directly related to {topic}. Do not include any introduction, explanation, or extra text. Only generate the question itself.")
                bot_response = generate_response(f"Generate a unique one-liner question related to the {topic} programming language. Do not include any introduction, explanation, or extra text. Only generate the question itself and avoid asking questions from this list: {asked_questions}."
)

        play_voice_response(bot_response)
        current_question = bot_response
        questions_attempted += 1
        questions_remaining -= 1
        count += 1
        return jsonify({'reply': bot_response})

    # Step 4: End the Interview after 10 questions
    elif count > 11:
        reply = 'Your interview has been completed successfully.'
        play_voice_response(reply)
        return jsonify({'reply': reply})

    # Default case
    return jsonify({'reply': 'Error: Unexpected case encountered.'})

def detect_emotions():
    """Scheduled function to detect emotions every 30 seconds."""
    global emotion_result
    global list_emotions
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Analyze the first detected face
        face_roi = frame[y:y + h, x:x + w]

        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            with lock:
                emotion_result = result.get('dominant_emotion', 'Unknown')
                list_emotions.append(emotion_result)
        except Exception as e:
            print(f"Error in scheduled emotion detection: {e}")

    cap.release()

@app.route('/checking', methods=['GET', 'POST'])
def checking():
    global selected_language
    global topic
    if request.method == 'POST':
        selected_language = request.form.get('language')
        topic = selected_language
    print("Selected Topic:", topic)
    return render_template('checking.html', language=selected_language)

@app.route('/')
def index():
    return render_template('FirstPage.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/instructions', methods=['POST'])
def instructions():
    global candidate_name
    global selected_language
    global topic
    if request.method == 'POST':
        candidate_name = request.form.get('name')
        selected_language = request.form.get('language')
    topic = selected_language
    return render_template('instructions.html')

@app.route('/interview' , methods = ['POST'])
def interview():
    global selected_language
    global candidate_name
    global topic
    if request.method == 'POST':
        selected_language = request.form.get('language')
    topic = selected_language
    print("Selected Language:", selected_language)
    return render_template('interview_main.html', language=selected_language, candidate_name = candidate_name)


def most_frequent_emotion(emotions):
    # Count the occurrences of each emotion in the list
    emotion_count = Counter(emotions)
    
    # Find the maximum count
    max_count = max(emotion_count.values())
    
    # Get all emotions that have the max count
    most_common_emotions = [emotion for emotion, count in emotion_count.items() if count == max_count]
    
    # If there's a tie, return 'neutral', else return the most frequent emotion
    if len(most_common_emotions) > 1:
        return 'neutral'
    else:
        return most_common_emotions[0]

@app.route('/analysis')
def analysis():

    global topic
    global final_emotion
    global wrong_questions
    global correct_questions
    global total_score_percentage
    global candidate_name

    response_data = ''
    bot_response = ''
    max_retries = 3  # Maximum number of retries for handling errors
    retries = 0

    prompt = (
     f"I have a dictionary containing question-answer pairs: {questions_answers}. For each question, evaluate the accuracy of the provided answers on a scale of 1 to 10 based on correctness and relevance. Include only improvement suggestions for answers that score less than 7. Calculate the total score and the overall percentage, then return a JSON response only with the structure:\n\n"
    "{\n"
    "  'scores': {\n"
    "    'Question 1': score,\n"
    "    'Question 2': score,\n"
    "    ...\n"
    "  },\n"
    "  'improvement': {\n"
    "    'Question with low score': 'Suggested feedback',\n"
    "    ...\n"
    "  },\n"
    "  'total_score': total,\n"
    "  'percentage': overall_percentage,\n"
    "  'correct_questions_count': correct_count,\n"
    "  'wrong_questions_count': wrong_count,\n"
    "  'area_of_improvement': 'Areas where improvements can be made'\n"
    "},generate only json response dont add any introduction part just return json structure start with  this '{' and end with this '}' and return only json response "
)


    # Try generating a response, retry if a 404 error occurs
    while retries < max_retries:
        bot_response = generate_response(prompt)
        if bot_response != 'Request ended with status code 404':
            break
        retries += 1

    # If no valid response is received after retries, return an error message
    if bot_response == 'Request ended with status code 404':
        return "Error: Unable to get a valid response from the bot.", 500

    # Parse the response JSON safely
    try:
        response_data = json.loads(bot_response)
        print (response_data)
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        return "Error: Failed to parse response from the bot.", 500

    # Extract data from the response
    correct_questions = response_data.get("correct_questions_count",0)
    wrong_questions = response_data.get("wrong_questions_count",0)
    area_of_improvement = response_data.get("area_of_improvement",{})
    total_score_percentage = response_data.get("percentage",0)
    # Determine the most frequent emotion if needed
    final_emotion = most_frequent_emotion(list_emotions)
    current_date = datetime.now().date()
    current_time = datetime.now().strftime("%H:%M:%S")

    # Pass the extracted data to the template for rendering
    return render_template(
        'analysis.html',
        candidate_name=candidate_name,
        topic=topic,
        correct_questions=correct_questions,
        wrong_questions=wrong_questions,
        total_score_percentage=total_score_percentage,
        final_emotion = final_emotion,
        area_of_improvement = area_of_improvement,
        current_date = current_date,
        current_time = current_time
    )

def generate_video_feed():
    global face_count, emotion_result

    cap = cv2.VideoCapture(0)
    # Scheduler setup
    scheduler = BackgroundScheduler()
    scheduler.add_job(detect_emotions, 'interval', seconds=30)
    scheduler.start()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        face_count = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Persons: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            face_roi = frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                emotion = result.get('dominant_emotion', 'Unknown')
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error in live emotion detection: {e}")

        # Display the last detected emotion from the scheduler
        with lock:
            if emotion_result:
                cv2.putText(frame, f'Scheduled Emotion: {emotion_result}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break

        frame_data = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    cap.release()

@app.route('/person_count')
def person_count():
    """API endpoint to get the current face count."""
    return jsonify({'count': face_count , 'confidence':confidence_score , 'attempted':questions_attempted , 'remaining':questions_remaining, 'cheating':cheating_detected})

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

