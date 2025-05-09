# Face-Detection-and-Recognition:
Here’s a simple and informative **README.md** file for your **Face Recognition** project:

---

## **Face Recognition using OpenCV and KNN**

This project implements a face recognition system using **OpenCV** and **K-Nearest Neighbors (KNN)**. It captures real-time face data through a webcam, trains a model to recognize faces, and identifies the faces during real-time video streaming.

---

### **Features**

* Real-time face detection and recognition.
* Uses KNN for face classification.
* Captures face data through a webcam.
* Supports multiple users with labeled face data.
* Stores training data as `.npy` files.

---

### **Technologies Used**

* Python
* OpenCV
* NumPy
* K-Nearest Neighbors (KNN)

---

### **Setup and Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/divyaa2003/Face-Recognition.git
   cd Face-Recognition
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Haar Cascade file:**
   Make sure you have the `haarcascade_frontalface_alt.xml` file in the project directory.
   You can download it from:
   [OpenCV GitHub Repository](https://github.com/opencv/opencv/tree/master/data/haarcascades)

---

### **Data Generation**

To collect face data for training:

```bash
python3 realtime_data_generation.py
```

* Enter the name of the person when prompted.
* It will capture face images and save them as `.npy` files in the `data` folder.

---

### **Face Recognition**

To recognize faces using the trained model:

```bash
python3 face_recognition_knn.py
```

* The script will detect and recognize faces from the webcam feed.

---

### **Troubleshooting**

* **Python Command Not Found:**
  Use `python3` instead of `python`.
* **Haar Cascade File Missing:**
  Make sure `haarcascade_frontalface_alt.xml` is in the current directory.
* **OpenCV Error - Size Width/Height > 0:**
  Check that the camera is working correctly and that the face is detected.

---




### **License**

This project is licensed under the MIT License.

---

Let me know if you need any changes or additional sections! 😊
