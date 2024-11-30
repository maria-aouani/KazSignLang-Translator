from PIL import Image, ImageDraw, ImageFont  # Import Pillow for text rendering
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO
from torch.nn.utils.rnn import pad_sequence

# Load YOLO model
yolo_model = YOLO(
    '/runs/pose/train5/weights/best.pt'
)

# Define Vocabulary class and load it with data
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def encode(self, sentence):
        return [self.word2idx.get(word, 0) for word in sentence.split()]  # 0 for unknown words

    def decode(self, indices):
        return [self.idx2word.get(idx, "") for idx in indices]

    def __len__(self):
        return len(self.word2idx) + 1  # +1 to account for padding token

# Initialize vocabulary
vocab = Vocabulary()
data = [
    "Әже Әже Әже қалпак қалпак ұмыту ұмыту", "Әже қалпақты ұмытты",
    "Әже қалпақты жақсы көрді", "Әже қалпағын жақсы көрді", "Қалпақты жақсы көрем",
    "Әже ұмытты", "Әже атын ұмытты", "Салеметсіз бе әже"
]
for sentence in data:
    vocab.add_sentence(sentence)

# Define and load Seq2Seq model
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Seq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_seq, target_seq):
        embedded_input = self.embedding(input_seq)
        _, (h, c) = self.encoder_lstm(embedded_input)

        embedded_target = self.embedding(target_seq)
        decoder_output, _ = self.decoder_lstm(embedded_target, (h, c))
        output = self.fc(decoder_output)
        return output

# Model parameters
embedding_dim = 64
hidden_dim = 100
vocab_size = len(vocab)
model = Seq2SeqModel(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('seq2seq_text_model.pth'))
model.eval()

# Initialize deque for gesture sequence
gesture_sequence = deque(maxlen=20)

# Load a font that supports Kazakh characters
font_path = "/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediaPipe_Yolo_Lstm_222/arial.ttf"  # Replace with the path to a suitable font
font = ImageFont.truetype(font_path, 15)  # Adjust font size as needed

# Function to draw text with PIL
def draw_text_with_pil(image, text, position, font, text_color=(255, 255, 255)):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Video capture setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for YOLO
    results = yolo_model(frame)
    writable_frame = frame.copy()

    # Process detections
    for result in results:
        annotated_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                gesture_class = box.cls  # Class index
                gesture_name = yolo_model.names[int(gesture_class)]  # Get gesture name
                gesture_sequence.append(gesture_name.lower())
                print("Detected gesture:", gesture_name.lower())

    # Process gesture_sequence with the Seq2Seq LSTM model
    if len(gesture_sequence) > 1:
        encoded_sequence = torch.tensor(vocab.encode(' '.join(gesture_sequence))).unsqueeze(0).long()

        # Prepare input for decoder (adding padding to match expected input length)
        target_sequence = torch.zeros_like(encoded_sequence)

        with torch.no_grad():
            output = model(encoded_sequence, target_sequence)
            predicted_indices = output.argmax(dim=2).squeeze().tolist()
            predicted_sentence = vocab.decode(predicted_indices)

        # Display the predicted sentence on the video feed
        sentence_text = ' '.join(predicted_sentence).strip()
        writable_frame = draw_text_with_pil(writable_frame, sentence_text, (3, 30), font, text_color=(255, 255, 255))

    # Display frame
    cv2.imshow('YOLOv8 Hand Keypoint Detection', writable_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
