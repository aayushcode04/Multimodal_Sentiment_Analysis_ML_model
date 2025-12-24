# 🎭 Multimodal Emotion Analysis System

A complete **end-to-end multimodal machine learning system** that analyzes **emotion and mood** by combining  
**text**, **speech (audio)**, and **facial expressions** into a single final score.

This project was built from scratch to understand **how real ML systems behave**, not just how to train a model.

---

## 🔍 What This Project Does

The system captures real-time input from a user and performs:

### 1️⃣ Text Emotion Analysis
- Speech is converted to text
- Text is passed through a trained deep learning model
- Outputs a sentiment score from **1 (very low)** to **5 (very high)**

### 2️⃣ Audio Emotion Analysis
- Voice energy and pitch are extracted
- Detects calm, neutral, or low/high emotional intensity
- Produces an independent audio-based score

### 3️⃣ Face Emotion Analysis
- Facial expressions are analyzed across video frames
- Produces an average face-based emotion score

### 4️⃣ Multimodal Fusion
- Text, audio, and face scores are combined
- Final score reflects **realistic human judgment**
- Avoids extreme or biased predictions

---


## 📊 Example Behavior

- Happy speech → score ~4–5  
- Neutral / average day → score ~3  
- Low / sad mood → score ~1-2  
- Mixed emotions are handled smoothly  

The system does **not blindly trust one modality**, similar to how humans judge emotions.

---

## ⚙️ Tech Stack

- Python
- TensorFlow / Keras (Text Model)
- OpenCV (Face Analysis)
- Librosa / PyAudio (Audio Features)
- NumPy, Pandas

---

## 🚀 How It Works (High-Level)

1. Capture user speech and video input
2. Analyze text, audio, and face independently
3. Combine all signals using fusion logic
4. Display final emotion score and mood

---


## 📁 Models & Dataset

To keep the repository lightweight:
- Trained models (`.h5`, `.keras`, `.pkl`)
- Large datasets

are **not included**.

If you need:
- trained models
- balanced dataset
- experiment details

feel free to message me on **LinkedIn** : https://www.linkedin.com/in/aayush-agrawal-somaiya04/.

---

## 👤 Author : Aayush 

Built independently to deeply understand **machine learning pipelines, data behavior, and system-level design**,  
without copying any tutorials .
