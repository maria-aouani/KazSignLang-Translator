import os
import cv2
import mediapipe as mp
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import shutil

# Paths and parameters
DATA_PATH = '/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediapipeDataset_YOLO_Lstm/MP_Data'
actions = np.array(['Name', 'ILoveYou', 'Hello', 'grandma', 'forget', 'kalpak'])

no_sequences = 30  # Number of sequences per action
sequence_length = 30  # Number of frames per sequence
num_keypoints = 21  # Number of keypoints for the hand

# Create train and validation directories
train_dir = os.path.join(DATA_PATH, 'train')
val_dir = os.path.join(DATA_PATH, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'labels'), exist_ok=True)

# Directories for storing data
images_dir = '/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediapipeDataset_YOLO_Lstm/MP_Data/train/images'
labels_dir = '/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediapipeDataset_YOLO_Lstm/MP_Data/train/labels'

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)


# Function to calculate bounding box from keypoints
def calculate_bbox(landmarks, image_width, image_height):
    x_coords = [landmark.x * image_width for landmark in landmarks]
    y_coords = [landmark.y * image_height for landmark in landmarks]

    # Find the min and max points for the bounding box
    bbox_xmin = min(x_coords)
    bbox_ymin = min(y_coords)
    bbox_xmax = max(x_coords)
    bbox_ymax = max(y_coords)

    # Return the bounding box in pixel values
    return bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax


# Function to save keypoints and bbox as labels in YOLO format for exactly two hands
def save_keypoints_as_label(hand_landmarks_list, label_file_path, action_label, image_width, image_height):
    # Ensure the label directory exists
    os.makedirs(os.path.dirname(label_file_path), exist_ok=True)

    # We only consider up to 2 hands for labeling
    num_hands = min(2, len(hand_landmarks_list))  # Limit to 2 hands

    with open(label_file_path, 'w') as f:
        for i in range(num_hands):  # Process exactly two hands
            hand_landmarks = hand_landmarks_list[i]

            # Calculate the bounding box for the current hand
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = calculate_bbox(hand_landmarks.landmark, image_width,
                                                                        image_height)

            # Calculate the center and dimensions of the bounding box
            bbox_x_center = (bbox_xmin + bbox_xmax) / 2 / image_width  # Normalize to image width
            bbox_y_center = (bbox_ymin + bbox_ymax) / 2 / image_height  # Normalize to image height
            bbox_width = (bbox_xmax - bbox_xmin) / image_width  # Normalize width
            bbox_height = (bbox_ymax - bbox_ymin) / image_height  # Normalize height

            # Write class ID and bounding box for the current hand in YOLO format
            f.write(f"{action_label} {bbox_x_center} {bbox_y_center} {bbox_width} {bbox_height}")

            # Write the keypoints (normalized x, y coordinates)
            for landmark in hand_landmarks.landmark:
                f.write(f" {landmark.x} {landmark.y}")

            # Move to the next line for the next hand's information
            f.write("\n")

# Function to create the YAML file for YOLOv8
def create_yaml_file(train_path, val_path, num_classes, class_names):
    yaml_data = {
        'train': train_path,
        'val': val_path,
        'nc': num_classes,  # number of classes
        'names': class_names.tolist()  # class names
    }
    yaml_file_path = os.path.join(DATA_PATH, 'kazakh_sign_language.yaml')

    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    print(f"YAML file created: {yaml_file_path}")

# Loop over each action
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture image")
                    continue

                # Convert the BGR image to RGB and process it with MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(frame_rgb)

                # If hand landmarks are detected
                if result.multi_hand_landmarks:
                    hand_landmarks_list = result.multi_hand_landmarks

                    # Save image
                    image_file_path = os.path.join(images_dir, f"{action}{sequence}{frame_num}.jpg")
                    cv2.imwrite(image_file_path, frame)

                    # Save labels (bounding boxes and keypoints for both hands in the same file)
                    label_file_path = os.path.join(labels_dir, f"{action}{sequence}{frame_num}.txt")
                    save_keypoints_as_label(hand_landmarks_list, label_file_path, actions.tolist().index(action),
                                            frame.shape[1], frame.shape[0])

                    # Draw the bounding box for each hand
                    for hand_landmarks in hand_landmarks_list:
                        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = calculate_bbox(hand_landmarks.landmark, frame.shape[1], frame.shape[0])
                        cv2.rectangle(frame, (int(bbox_xmin * frame.shape[1]), int(bbox_ymin * frame.shape[0])),
                                      (int(bbox_xmax * frame.shape[1]), int(bbox_ymax * frame.shape[0])), (0, 255, 0), 2)

                # Overlay the current action being recorded
                cv2.putText(frame, f"Recording: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                # Display the frame with annotations (optional)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('MediaPipe Feed', frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()

# Split data into train and val sets
train_images, val_images = train_test_split(os.listdir(images_dir), test_size=0.2, random_state=42)

# Move files to respective train/val folders
for image_file in train_images:
    shutil.move(os.path.join(images_dir, image_file), os.path.join(train_dir, 'images', image_file))
    label_file = image_file.replace('.jpg', '.txt')
    shutil.move(os.path.join(labels_dir, label_file), os.path.join(train_dir, 'labels', label_file))

for image_file in val_images:
    shutil.move(os.path.join(images_dir, image_file), os.path.join(val_dir, 'images', image_file))
    label_file = image_file.replace('.jpg', '.txt')
    shutil.move(os.path.join(labels_dir, label_file), os.path.join(val_dir, 'labels', label_file))

# Create YAML file
create_yaml_file(train_path=train_dir, val_path=val_dir, num_classes=len(actions), class_names=actions)