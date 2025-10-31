const socket = io();
let localStream = null;
let roomId = '';
let userId = '';
let username = '';

document.getElementById('create-room-btn').onclick = async () => {
  const res = await axios.post('/api/create_meeting');
  document.getElementById('room-id').value = res.data.room_id;
};

document.getElementById('join-btn').onclick = () => {
  roomId = document.getElementById('room-id').value;
  username = document.getElementById('username').value || 'Anonymous';

  socket.emit('join_meeting', { room_id: roomId, username });
  document.getElementById('join-section').classList.add('hidden');
  document.getElementById('meeting-section').classList.remove('hidden');
  document.getElementById('room-info').innerText = `Room: ${roomId}`;
};

document.getElementById('leave-btn').onclick = () => {
  socket.emit('leave_meeting');
  document.getElementById('meeting-section').classList.add('hidden');
  document.getElementById('join-section').classList.remove('hidden');
  if (localStream) {
    localStream.getTracks().forEach(track => track.stop());
  }
};

document.getElementById('start-video').onclick = async () => {
  const video = document.getElementById('video-preview');
  localStream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = localStream;

  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const sendFrame = () => {
    if (!localStream) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg');
    socket.emit('video_frame', { frame: dataURL });
    requestAnimationFrame(sendFrame);
  };
  sendFrame();
};

document.getElementById('stop-video').onclick = () => {
  if (localStream) {
    localStream.getTracks().forEach(track => track.stop());
    localStream = null;
  }
};

document.getElementById('send-text-btn').onclick = () => {
  const text = document.getElementById('text-input').value;
  if (text) {
    socket.emit('text_message', { text });
  }
};

socket.on('joined_meeting', data => {
  userId = data.user_id;
});

socket.on('sign_translation', data => {
  appendOutput(`[SIGN] ${data.username || data.user_id}: ${data.word} (${data.confidence})`);
});

socket.on('speech_translation', data => {
  appendOutput(`[SPEECH] ${data.text}`);
  if (data.video_url) {
    const vid = document.createElement('video');
    vid.src = data.video_url;
    vid.controls = true;
    vid.className = 'w-full rounded';
    document.getElementById('output').appendChild(vid);
  }
});

socket.on('text_to_sign', data => {
  appendOutput(`[TEXT] ${data.text}`);
  if (data.video_url) {
    const vid = document.createElement('video');
    vid.src = data.video_url;
    vid.controls = true;
    vid.className = 'w-full rounded';
    document.getElementById('output').appendChild(vid);
  }
});

function appendOutput(text) {
  const div = document.createElement('div');
  div.innerText = text;
  document.getElementById('output').appendChild(div);
}
