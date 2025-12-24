import cv2
import numpy as np
import time
import pickle
import threading
import pyaudio
import wave
import speech_recognition as sr
import librosa
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# SIMPLE MOOD ANALYSIS SYSTEM - FIXED VERSION
# =============================================================================

# Load models once 
print("Loading models...")
face_model = load_model("C:/Users/Admin/Downloads/mood_model.h5")
text_model = load_model("C:/Users/Admin/Downloads/new_sentiment_model.h5")
with open("C:/Users/Admin/Downloads/new_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Settings
MOOD_LABELS = ["Very Bad", "Bad", "Neutral", "Happy", "Very Happy"]
all_face_scores = []  # Store ALL scores for final average
recent_face_scores = deque(maxlen=30)  # For moving average
audio_scores = []
recording = False
audio_frames = []

def get_face_score(gray):
    """Get face expression score (fixed to be neutral by default)"""
    try:
        # Model prediction
        img = cv2.resize(gray, (64, 64))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        pred = face_model.predict(img, verbose=0)
        
        # Base score from model
        base_score = float(np.argmax(pred[0]) + 1)
        
        # Adjust to be neutral (3-3.5) by default
        if base_score == 3:
            score = 3.2  # Slightly above neutral
        elif base_score == 4:
            score = 3.8  # Happy but not too high
        elif base_score == 5:
            score = 4.2  # Very happy
        elif base_score == 2:
            score = 2.8  # Bad but not terrible
        else:
            score = 2.0  # Very bad
            
        return score
    except:
        return 3.0

def detect_smile(gray):
    """Detect smile with confidence"""
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 25, minSize=(25, 15))
        
        if len(smiles) > 0:
            # Calculate smile confidence
            smile_areas = [s[2]*s[3] for s in smiles]
            avg_area = np.mean(smile_areas) if smile_areas else 0
            
            if avg_area > 500:  # Big smile
                return 4.8, True
            elif avg_area > 200:  # Normal smile
                return 4.3, True
            else:  # Small smile
                return 4.0, True
    
    return 0, False

def analyze_face(frame):
    """Main face analysis with proper scoring"""
    global all_face_scores
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Get base expression
    base_score = get_face_score(gray)
    
    # Check for smile
    smile_score, has_smile = detect_smile(gray)
    
    # Apply smile boost if detected
    if has_smile:
        final_score = max(base_score, smile_score)  # Smile takes priority
    else:
        final_score = base_score
    
    # Store scores (BOTH lists)
    all_face_scores.append(final_score)  # Store ALL scores
    recent_face_scores.append(final_score)  # Store recent for moving average
    
    # Display
    mood_idx = min(4, max(0, int(final_score) - 1))
    mood_text = MOOD_LABELS[mood_idx]
    
    # Color based on mood
    if final_score >= 4:
        color = (0, 255, 0)  # Green
    elif final_score >= 3:
        color = (0, 255, 255)  # Yellow
    elif final_score >= 2:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 0, 255)  # Red
    
    cv2.putText(frame, f"Mood: {mood_text} ({final_score:.1f}/5)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show moving average (from recent scores)
    if recent_face_scores:
        moving_avg = np.mean(recent_face_scores)
        cv2.putText(frame, f"Moving Avg: {moving_avg:.1f}/5", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Smile indicator
    if has_smile:
        cv2.putText(frame, "SMILE", (300, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame

def record_audio():
    """Record audio in background"""
    global recording, audio_frames
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                   rate=44100, input=True,
                   frames_per_buffer=1024)
    
    print("\n[RECORDING STARTED] Speak now...")
    audio_frames = []
    
    while recording:
        data = stream.read(1024, exception_on_overflow=False)
        audio_frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("[RECORDING STOPPED]")

def analyze_speech():
    """Analyze recorded speech - FIXED to show ALL details in terminal"""
    global audio_frames
    
    if not audio_frames:
        print("No audio recorded!")
        return 3.0, ""
    
    try:
        # Save audio to file
        filename = "temp_speech.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(audio_frames))
        wf.close()
        
        # Convert speech to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        
        # FIX 2: Show speech text clearly in terminal
        print("\n" + "="*50)
        print("SPEECH ANALYSIS")
        print("="*50)
        print(f"You said: {text}")
        
        # Analyze text sentiment
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=90, padding='post')
        pred = text_model.predict(padded, verbose=0)
        text_score = float(np.argmax(pred[0]) + 1)
        print(f"Text sentiment score: {text_score:.1f}/5")
        
        # Simple pitch analysis
        y, sr_audio = librosa.load(filename, sr=44100)
        
        # FIX 3: Show pitch analysis details
        # Check if audio has enough energy
        energy = np.mean(np.abs(y))
        print(f"Voice energy: {energy:.4f}")
        
        if energy < 0.01:  # Too quiet
            pitch_score = 3.0
            print("Pitch: Too quiet (default: 3.0/5)")
        else:
            # Simple pitch detection
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr_audio)
            
            # Find pitches with significant magnitude
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0 and magnitudes[index, t] > np.mean(magnitudes):
                    pitch_values.append(pitch)
            
            if pitch_values:
                avg_pitch = np.mean(pitch_values)
                print(f"Average pitch: {avg_pitch:.1f} Hz")
                
                if 120 <= avg_pitch <= 280:
                    if avg_pitch > 200:
                        pitch_score = 4.0
                        print("Pitch: High/energetic (4.0/5)")
                    else:
                        pitch_score = 3.5
                        print("Pitch: Normal (3.5/5)")
                else:
                    pitch_score = 3.0
                    print(f"Pitch: Outside normal range (3.0/5)")
            else:
                pitch_score = 3.0
                print("Pitch: No clear pitch detected (3.0/5)")
        
        # Combined score (70% text, 30% pitch)
        audio_score = (text_score * 0.7 + pitch_score * 0.3)
        print(f"\nCombined audio score: {audio_score:.1f}/5")
        print("="*50)
        
        # Store score
        audio_scores.append(audio_score)
        
        return audio_score, text
        
    except sr.UnknownValueError:
        print("\n[ERROR] Could not understand speech")
        return 3.0, ""
    except sr.RequestError as e:
        print(f"\n[ERROR] Speech recognition error: {e}")
        return 3.0, ""
    except Exception as e:
        print(f"\n[ERROR] Speech analysis failed: {e}")
        return 3.0, ""

def main():
    """Main program"""
    global recording, all_face_scores
    
    print("\n" + "="*50)
    print("MOOD ANALYSIS SYSTEM - FIXED VERSION")
    print("="*50)
    print("Press 's' to start/stop speaking")
    print("Press 'q' to quit")
    print("="*50)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return
    
    recording_thread = None
    current_text = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze face
        frame = analyze_face(frame)
        
        # Display instructions
        if recording:
            # Blinking recording indicator
            blink = int(time.time() * 2) % 2
            if blink:
                cv2.putText(frame, "● RECORDING - Speak now!", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 's' to speak", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show last speech
        if current_text:
            short_text = current_text[:35] + "..." if len(current_text) > 35 else current_text
            cv2.putText(frame, f"Last: {short_text}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # FIX 1: Show proper averages
        if all_face_scores:  # Use ALL face scores for overall average
            overall_face_avg = np.mean(all_face_scores)
            cv2.putText(frame, f"Overall Face: {overall_face_avg:.1f}/5", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        if audio_scores:
            audio_avg = np.mean(audio_scores)
            cv2.putText(frame, f"Audio Avg: {audio_avg:.1f}/5", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            
            # Combined average
            if all_face_scores:
                total = (overall_face_avg + audio_avg) / 2
                cv2.putText(frame, f"Overall Total: {total:.1f}/5", (10, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Mood Analysis - Press S to speak, Q to quit", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            if not recording:
                # Start recording
                recording = True
                recording_thread = threading.Thread(target=record_audio, daemon=True)
                recording_thread.start()
            else:
                # Stop recording
                recording = False
                if recording_thread:
                    recording_thread.join(timeout=1)
                
                # Analyze speech
                print("\nAnalyzing speech...")
                score, text = analyze_speech()
                if text:
                    current_text = text
    
    # Cleanup
    recording = False
    cap.release()
    cv2.destroyAllWindows()
    
    # Show final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    if all_face_scores:
        face_avg = np.mean(all_face_scores)  # FIX 1: Use ALL scores
        face_mood = MOOD_LABELS[min(4, max(0, int(face_avg)-1))]
        print(f"Face Analysis:")
        print(f"  Total frames: {len(all_face_scores)}")
        print(f"  Average score: {face_avg:.2f}/5")
        print(f"  Mood: {face_mood}")
    
    if audio_scores:
        audio_avg = np.mean(audio_scores)
        audio_mood = MOOD_LABELS[min(4, max(0, int(audio_avg)-1))]
        print(f"\nAudio Analysis:")
        print(f"  Speech samples: {len(audio_scores)}")
        print(f"  Average score: {audio_avg:.2f}/5")
        print(f"  Mood: {audio_mood}")
        
        if all_face_scores:
            total = (face_avg + audio_avg) / 2
            total_mood = MOOD_LABELS[min(4, max(0, int(total)-1))]
            print(f"\nOverall Result:")
            print(f"  Combined score: {total:.2f}/5")
            print(f"  Overall mood: {total_mood}")
    
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

