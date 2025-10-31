import socketio
import time
import json

# Create a Socket.IO client
sio = socketio.Client()

# Connect to the server
sio.connect('http://localhost:5000')

# Event handlers for testing
@sio.event
def connect():
    print("Connected to server")
    
@sio.event
def disconnect():
    print("Disconnected from server")

@sio.on('meeting_created')
def on_meeting_created(data):
    print(f"Meeting created: {json.dumps(data, indent=2)}")

@sio.on('meeting_joined')
def on_meeting_joined(data):
    print(f"Meeting joined: {json.dumps(data, indent=2)}")

@sio.on('live_translation')
def on_live_translation(data):
    print(f"Live translation: {json.dumps(data, indent=2)}")

# Test create meeting
print("\nTesting create meeting...")
sio.emit('create_meeting', {
    'username': 'TestUser',
    'user_type': 'speaker'
})

time.sleep(2)  # Wait for response

# Test join meeting
print("\nTesting join meeting...")
sio.emit('join_meeting', {
    'room_id': 'test-room',
    'username': 'TestUser2',
    'user_type': 'non_speaker'
})

time.sleep(2)  # Wait for response

# Test live speech to sign
print("\nTesting live speech to sign...")
sio.emit('live_speech_to_sign', {
    'room_id': 'test-room',
    'text': 'Hello world'
})

time.sleep(2)  # Wait for response

# Test live sign translation
print("\nTesting live sign translation...")
sio.emit('live_sign_translation', {
    'room_id': 'test-room',
    'frame_data': 'base64_encoded_frame_data'
})

time.sleep(2)  # Wait for response

# Disconnect
print("\nDisconnecting...")
sio.disconnect()
