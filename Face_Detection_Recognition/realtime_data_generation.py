import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)

# Detecting Face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'

# Taking File Name as Input from the User
file_name = input("Enter the name of the person: ")

max_images = 100  # Maximum number of images to capture

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    if len(faces) == 0:
        print("No faces detected.")
        continue

    # Sort faces based on the area (pick the largest one)
    faces = sorted(faces, key=lambda f: f[2] * f[3])

    # Loop through detected faces (taking the largest one)
    for face in faces[-1:]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Crop the face with an offset
        offset = 10
        x1, y1 = max(0, x - offset), max(0, y - offset)
        x2, y2 = min(frame.shape[1], x + w + offset), min(frame.shape[0], y + h + offset)
        face_section = frame[y1:y2, x1:x2]

        # Validate the cropped face section
        if face_section.size == 0 or face_section.shape[0] == 0 or face_section.shape[1] == 0:
            print("Warning: Invalid face crop, skipping this frame.")
            continue

        # Resize the face section to 100x100
        face_section = cv2.resize(face_section, (100, 100))

        skip += 1

        # Store every 10th frame
        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"Captured {len(face_data)} faces...")

    # Show the frame and face section if valid
    cv2.imshow("Frame", frame)

    if face_section.size != 0:
        cv2.imshow("Face Section", face_section)

    # Check if the maximum number of images is reached
    if len(face_data) >= max_images:
        print(f"Captured {max_images} faces. Stopping...")
        break

    # Stop the loop if 'q' is pressed
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Convert the face list array into a numpy array and save it
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
np.save(dataset_path + file_name + '.npy', face_data)
print(f"Data successfully saved at {dataset_path}{file_name}.npy")

cap.release()
cv2.destroyAllWindows()
