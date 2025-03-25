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
import re
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
list_emotions = ['neutral']
count = 0
confidence_score = 50
questions_attempted = 0
questions_remaining = 11  # Initial number of questions
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
bot = True
# @app.route('/play_voice_response', methods = ['GET','POST'])
def play_voice_response(text):
    global bot
    """Generate and play a voice response."""
    # Remove non-alphanumeric characters except spaces
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    sound = gtts.gTTS(clean_text, lang='en')
    sound.save("response.mp3")
    bot = False
    playsound.playsound("response.mp3")
    bot = True

# Function to generate a response using g4f (GPT-4)
def generate_response(input_text):
    print("promt entered...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # You can adjust the model name here if needed
        messages=[{"role": "user", "content": input_text}],
    )
    print("generated......")
    return response.choices[0].message.content  # Extract the response text

@app.route('/submit_response', methods=['GET', 'POST'])
def submit_response():
    global topic, count, questions_attempted, questions_remaining, candidate_name, current_question
    global asked_questions
    global questions_answers
    data = request.json
    reply = ''
    bot_response = ''
    if count >= 2:
        questions_attempted += 1
        questions_remaining -= 1
    # Step 1: Generate Introduction Question
    if count == 0:
        reply = f"Hi {candidate_name}, Please introduce yourself."
        play_voice_response(reply)
        current_question = reply
        count += 1
        return jsonify({'reply': reply})
        # reply = 'Your interview has been completed successfully.'
        # print("wl to debug")
        # play_voice_response(reply)
        # analysis = "view_analysis"
        # return jsonify({'reply': reply, 'analysis':analysis})
    
    # Step 2: Generate Topic-Specific Question
    elif count == 1:
        reply = f"As you have chosen {topic}, I will be asking questions on the {topic} programming language."
        play_voice_response(reply)
        bot_response = generate_response(
    f"Generate a unique one-liner question related to the {topic} programming language. "
    f"Only generate the question itself—no answers, explanations, or extra text. "
    f"Do not repeat questions from this list: {asked_questions}. Ensure the question is relevant, concise, and in English."
)
        print("bot_response in 1--->",bot_response,"<----bot_response")
        while bot_response == 'Request ended with status code 404' or bot_response == 'Request ended with status code 403' or bot_response is None or bot_response == '':
                bot_response = generate_response(
    f"Generate a unique one-liner question related to the {topic} programming language. "
    f"Only generate the question itself—no answers, explanations, or extra text. "
    f"Do not repeat questions from this list: {asked_questions}. Ensure the question is relevant, concise, and in English."
)
        
        if bot_response and bot_response not in ['Request ended with status code 404', 'Request ended with status code 403']:
            try:
                play_voice_response(bot_response)  # Only call this if the response is valid
            except Exception as e:
                print(f"Error playing voice response: {e}")
        else:
            print("Failed to generate a valid bot response after retries.")
        current_question = bot_response
        asked_questions.append(current_question)
        questions_answers[current_question] = ''  # Initialize the dictionary with an empty answer
        count += 1
        return jsonify({'reply': bot_response})

    elif count > 1 and count < 12:
        candidate_answer = data.get('response', '')
        if current_question:
            # Store the answer for the current question
            asked_questions.append(current_question)
            print(f"Question: {current_question} | Answer: {candidate_answer}")
            questions_answers[current_question] = candidate_answer
            
        bot_response = generate_response(
    f"Generate a unique one-liner question related to the {topic} programming language. "
    f"Do not include any introductions, explanations, extra text, or answers. "
    f"Only output the question itself, without repeating questions from this list: {asked_questions}. "
    f"Ensure the response is strictly a single concise and relevant question in English."
)

        # Retry generating a valid response in case of errors or invalid outputs
        retry_count = 0
        max_retries = 3
        while not bot_response or bot_response in ['Request ended with status code 404', 
                                                'Request ended with status code 403']:
            bot_response = generate_response(
    f"Generate a unique one-liner question related to the {topic} programming language. "
    f"Do not include any introductions, explanations, extra text, or answers. "
    f"Only output the question itself, without repeating questions from this list: {asked_questions}. "
    f"Ensure the response is strictly a single concise and relevant question in English."
)
            retry_count += 1

        if bot_response and bot_response not in ['Request ended with status code 404', 
                                                'Request ended with status code 403']:
            try:
                play_voice_response(bot_response)  # Only call this if the response is valid
            except Exception as e:
                print(f"Error playing voice response: {e}")
        else:
            print("Failed to generate a valid bot response after retries.")

        current_question = bot_response

        count += 1

        return jsonify({'reply': bot_response})

    # Step 4: End the Interview after 10 questions
    elif count == 12:
        reply = 'Your interview has been completed successfully.'
        print("wl to debug")
        play_voice_response(reply)
        final_count = count
        count = 0
        analysis = "view_analysis"
        return jsonify({'reply': reply, 'count': final_count})
    # Default case
    return jsonify({'reply': 'Error: Unexpected case encountered.'})

@app.route('/stopInterview')
def stopInterview():
    return render_template('stopInterview.html')
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

def extract_json_response(input_string):
    try:
        # Find the start and end of the JSON block
        start_index = input_string.find('{')
        end_index = input_string.rfind('}')
        
        # Extract the JSON substring
        json_string = input_string[start_index:end_index + 1]
        
        # Parse and return the JSON as a dictionary
        return json.loads(json_string)
    except (ValueError, json.JSONDecodeError):

        return False
    
# @app.route('/questions_wise' , methods = ['GET'])
# def questions_wise():
#     print("hello0")
#     global questions_answers
#     # Generate the prompt
#     prompt = f"""
# I have a dictionary named `questions_answers` where each key is a question (string), and the value is the user's answer (string). Evaluate the correctness of each answer based on the question-answer pair. Use the following criteria:

# 1. If the answer is correct and relevant, classify the question as "Correct." Add it to an array with the format:
#    {{ "question": "Question text", "user_answer": "User's answer" }}.

# 2. If the answer is wrong or irrelevant, classify the question as "Wrong." Add it to a separate array with the format:
#    {{ "question": "Question text", "user_answer": "User's answer", "expected_answer": "Correct answer or explanation" }}.

# Return the results in the following JSON structure:
# {{
#   "correct_questions": [
#     {{ "question": "Question text", "user_answer": "User's answer" }},
#     ...
#   ],
#   "wrong_questions": [
#     {{ "question": "Question text", "user_answer": "User's answer", "expected_answer": "Correct answer or explanation" }},
#     ...
#   ]
# }}

# Here is the dictionary of question-answer pairs to analyze:
# {questions_answers}

# Ensure that your response is a valid JSON object starting with '{{' and ending with '}}'. Do not include any other text or explanation in your response.
# """

#     print("questions_answers",questions_answers)
#     bot_response = generate_response(prompt)
#     bot_response = extract_json_response(bot_response)
#     response_data = ''
#     if bot_response == False:
#         bot_response = {}
        
#     if isinstance(bot_response, dict):
#         bot_response = json.dumps(bot_response)
#     try:
#         response_data = json.loads(bot_response)
#     except json.JSONDecodeError as e:
#         print("JSON parsing error:", e)
#         return "Error: Failed to parse response from the bot.", 500
#     print(bot_response,"bot_response")
#     print(response_data,"response_data")
#     return render_template(
#         'questions_analysis.html',
#         correct_questions=bot_response['correct_questions'],
#         wrong_questions=bot_response['wrong_questions']
#     )

@app.route('/questions_wise', methods=['GET'])
def questions_wise():
    print("Starting questions_wise endpoint")
    global questions_answers

    # Arrays to hold results for correct and wrong questions
    correct_questions = []
    wrong_questions = []
    medium_questions = []
    # Iterate through each question-answer pair
    for question, user_answer in questions_answers.items():
        print(f"Processing Question: {question}")
        print(f"User Answer: {user_answer}")
        print("-" * 30)

        # Generate the prompt for GPT
        prompt = f"""
Evaluate the correctness of the following question and answer pair:

Question: "{question}"
User Answer: "{user_answer}"

Provide the result in the following JSON format:
{{
  "result": "Correct" or "Wrong" or "Medium",
  "expected_answer": "Correct answer or explanation"
}}

Ensure your response is a valid JSON object.
"""
        try:
            # Call your GPT model to generate a response
            bot_response = generate_response(prompt)  # Replace with actual GPT call
            print(f"GPT Response: {bot_response}")
            bot_response = extract_json_response(bot_response)
            if bot_response == False:
                bot_response = {}
                # /Users/apple/Documents/FinalAIProjectsWithDocumentations/AiMockInterview/app.py
            if isinstance(bot_response, dict):
                bot_response = json.dumps(bot_response)
            try:
                response_data = json.loads(bot_response)
            except json.JSONDecodeError as e:
                 print("JSON parsing error:", e)
                 return "Error: Failed to parse response from the bot.", 500
            # Process the GPT result
            if response_data.get("result") == "Correct":
                print("in correct")
                correct_questions.append({
                    "question": question,
                    "user_answer": user_answer
                })
            else:
                print("in wrong")
                wrong_questions.append({
                    "question": question,
                    "user_answer": user_answer,
                    "expected_answer": response_data.get("expected_answer", "No explanation provided")
                })

        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            continue
    print("correct_questions--->",correct_questions)
    print("wrong_questions--->",wrong_questions)
    # Render the results in the template
    return render_template(
        'question_wise.html',
        correct_questions=correct_questions,
        wrong_questions=wrong_questions
    )

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
    f"I have a dictionary containing question-answer pairs: {questions_answers}. For each question, evaluate the accuracy of the provided answers on a scale of 1 to 10 based on correctness and relevance. Provide improvement suggestions for answers scoring less than 7. Finally, calculate the total score, the overall percentage, the count of correct and incorrect questions, and suggest an area of improvement. Return only a valid JSON object with the exact structure:\n"
    "{\n"
    "  \"scores\": {\n"
    "    \"Question 1\": score,\n"
    "    \"Question 2\": score,\n"
    "    ...\n"
    "  },\n"
    "  \"improvement\": {\n"
    "    \"Question with low score\": \"Suggested feedback\",\n"
    "    ...\n"
    "  },\n"
    "  \"total_score\": total,\n"
    "  \"percentage\": overall_percentage,\n"
    "  \"correct_questions_count\": correct_count,\n"
    "  \"wrong_questions_count\": wrong_count,\n"
    "  \"area_of_improvement\": \"Areas where improvements can be made\"\n"
    "}\n"
    "Ensure the response starts with '{' and ends with '}'. Do not include any other text.your respose will start like this--> { and end with this }."
)

    bot_response = generate_response(prompt)

    if  bot_response in ['Request ended with status code 404', 
                                                'Request ended with status code 403' ,None, '']:
        
            while  bot_response in ['Request ended with status code 404', 
                                                        'Request ended with status code 403' ,None, '']:
                bot_response = generate_response(prompt)

    bot_response = extract_json_response(bot_response)
    if bot_response == False:
        bot_response = {}
        
    if isinstance(bot_response, dict):
        bot_response = json.dumps(bot_response)
    try:
        response_data = json.loads(bot_response)
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
    current_time = datetime.now().time()
    formatted_time = current_time.strftime("%H:%M:%S")
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
        current_time = formatted_time
    )

def generate_video_feed():
    global face_count, emotion_result

    cap = cv2.VideoCapture(0)
    # Scheduler setup
    scheduler = BackgroundScheduler()
    scheduler.add_job(detect_emotions, 'interval', seconds = 30)
    scheduler.add_job(check_confidence, 'interval', seconds = 10)
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

def check_confidence():
    global list_emotions
    global confidence_score
    # Define the confidence scores for each emotion
    emotion_scores = {
        "neutral": 50,
        "happy": 70,
        "sad": 30,
        "angry": 20
    }
    
    # Count the frequency of each emotion in the list
    emotion_counts = {emotion: list_emotions.count(emotion) for emotion in set(list_emotions)}
    
    # Find the most frequent emotion
    most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
    
    # Assign the confidence score based on the most frequent emotion
    confidence_score = emotion_scores.get(most_frequent_emotion, 0)  # Default score is 0 if emotion not in dict
    
    # print(f"Most frequent emotion: {most_frequent_emotion}, Confidence score: {confidence_score}")
    return confidence_score


@app.route('/person_count')
def person_count():
    global confidence_score
    global bot
    """API endpoint to get the current face count."""
    return jsonify({'count': face_count , 'confidence':confidence_score , 'attempted':questions_attempted , 'remaining':questions_remaining, 'cheating':cheating_detected , 'bot':bot})

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
