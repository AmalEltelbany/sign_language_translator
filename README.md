# Together: AI-Based Sign Language Translator

**Graduation Project – 2025**  
*Computer Engineering Department*

---

## Overview

**Together** is a multi-platform system designed to bridge communication between deaf and hearing individuals using **artificial intelligence**, **computer vision**, and **natural language processing**.

The project delivers real-time, bi-directional translation between **sign language**, **text**, and **speech**, enabling inclusive interaction in education, meetings, and daily life.  
It is available as both:
- **Mobile Application** built with **Flutter** (Android/iOS)  
- **Web Application** built with **HTML, CSS, and JavaScript**

---

## Motivation

Around the world, millions of people who are deaf or hard of hearing struggle to communicate effectively in environments where sign language is not widely understood.  
Current tools often handle only one translation direction (e.g., speech to sign) or require specialized hardware.

**Together** was developed to address these limitations by offering a **real-time, hardware-free, bi-directional translation solution** that supports **American Sign Language (ASL)** and can be extended to **Arabic Sign Language** in the future.

---

## Main Features

- **Sign to Speech / Text** – Recognizes sign gestures from a live camera feed and translates them into text or spoken words.  
- **Speech to Sign** – Converts spoken voice into corresponding sign language animations.  
- **Live Meeting Translation** – Enables real-time two-way translation in online meetings using **WebRTC**.  
- **Word to Sign Search** – Retrieves sign language videos for typed words.  
- **User Authentication** – Secure login, registration, and role-based access through **Firebase Authentication**.

---

## System Architecture

The system is composed of several integrated components:

- **Frontend (Flutter + Web)** – Provides the user interface and manages input from the camera and microphone.  
- **Backend (Flask API)** – Handles translation requests, model inference, and real-time communication.  
- **Machine Learning Engine (TensorFlow + MediaPipe)** – Performs gesture recognition and sequence prediction using a **Bidirectional LSTM (BiLSTM)** model.  
- **Database (Firebase Firestore & Storage)** – Stores user data, translation results, and sign videos.  
- **Real-Time Layer (Socket.IO + WebRTC)** – Manages live video and audio streaming for meetings.

---

## AI Model

- **Model Type:** Bidirectional LSTM  
- **Input:** 20-frame sequences (258 MediaPipe keypoints per frame)  
- **Accuracy:** ~94.8% on the test set  
- **Dataset:** Custom ASL dataset (50 words)  
- **Training Tools:** TensorFlow, NumPy, Google Colab  
- **Optimization:** Model quantization and TensorFlow Lite for mobile performance

---

## Technology Stack

| Layer | Technologies |
|-------|---------------|
| Mobile | Flutter (Dart) |
| Web | HTML, CSS, JavaScript |
| Backend | Python (Flask), Socket.IO |
| AI / ML | TensorFlow, MediaPipe |
| Database | Firebase Firestore, Firebase Storage |
| Real-Time Communication | WebRTC |
| Speech Processing | Google Speech-to-Text, gTTS |
| Tools | Google Colab, Git, TensorBoard |

---

## Evaluation

| Dataset | Accuracy | Precision | Recall | F1-Score |
|----------|-----------|------------|---------|-----------|
| Training | 100% | 93.8% | 93.9% | 93.8% |
| Validation | 94.98% | 94.7% | 94.8% | 94.75% |
| Test | 93.73% | 93.5% | 93.6% | 93.55% |

The system achieved **real-time performance with <1 second latency**, meeting all key functional and non-functional requirements.

---

## Security and Accessibility

- End-to-end encryption through **DTLS-SRTP** for media streams  
- Role-based access control and Firebase security rules  
- Accessible UI design with high contrast and large interaction areas

---

## Future Work

- Expand to **Arabic Sign Language (ArSL)**  
- Train transformer-based models (e.g., TimeSformer)  
- Add continuous sentence recognition  
- Develop offline mode for low-connectivity areas

---


**Supervisor:** Prof. Medhat Awadalla

---

## How to Run

### Backend
```bash
cd backend
pip install -r requirements.txt
python deploy.py
