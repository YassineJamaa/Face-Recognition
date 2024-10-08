# Face Recognition System

This project is inspired by the article *"Machine Learning is Fun! Part 4: Modern Face Recognition with Deep Learning"* by Adam Geitgey. You can read the original article on [Medium here](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78).

This real-time face recognition system processes video input from a webcam, detects faces, and stores face data in an SQLite database. The database contains the name (initially "Unknown"), face image, and face embeddings generated by a pre-trained FaceNet model. The system compares the embeddings of newly detected faces against existing entries to avoid duplicate storage and only saves new faces when necessary. To ensure smooth real-time video processing, the application leverages multithreading for parallel processing.

## Project Overview

The project focuses on efficiently storing facial information while processing several images per second. The goal is to minimize redundant data by checking whether a detected face already exists in the database using embedding distances. This allows the system to operate in real-time without excessive memory usage or database clutter.

### Key Features:
- Real-time face detection and alignment using the Dlib library.
- Face embedding generation using a pre-trained FaceNet model.
- Storage of face data (name, face image, and embeddings) in an SQLite database.
- Embedding distance comparison to check for existing faces, preventing duplicate entries.
- Graphical User Interface (GUI) for managing and labeling faces in the database.
- Parallel processing for smoother video stream handling and background image processing.

## Project Structure

### 1. `FaceEmbeddings` Class

This class handles the core face detection, alignment, and embedding extraction functionalities:

- **`import_image(image_path)`**: Loads and optionally resizes an image from the provided path.
- **`detect_face(image)`**: Detects faces in the input image using Dlib's frontal face detector.
- **`landmarks_face(image, bound_box)`**: Extracts facial landmarks for a detected face.
- **`face_encodings(image, landmarks_points)`**: Generates face embeddings from the detected face using the FaceNet model.
- **`align_face(image, img_dim, bound_box)`**: Aligns a detected face based on its landmarks.
- **`pipeline(image_path)`**: Full face processing pipeline that detects faces, extracts embeddings, aligns faces, and returns face information.

### 2. `PipelineDatabase` Class

This class integrates the face embeddings processing with the database management to avoid redundant entries:
- **`pipeline(image)`**: Runs the pipeline for detecting and processing faces from the webcam stream, checking if a face already exists in the database by comparing embeddings and storing new faces if necessary.

### 3. `DatasetManagerSQLite` Class

This class manages the SQLite database for storing face data:
- **`create_table()`**: Creates the database table if it doesn't exist.
- **`add_row(face_info)`**: Inserts a new face into the database with its name, embeddings, and face image.
- **`update_face_name(new_name, id)`**: Updates the name of a person based on their ID in the database.
- **`delete()`**: Deletes all entries from the database.

### 4. `FaceDatabaseGUI` Class

This class provides a graphical interface (using Tkinter) to display and manage the database. Users can:
- View faces stored in the database.
- Update names of "Unknown" faces.
- Filter to view only unknown faces.
- Go back to the webcam feed.

### 5. `WebcamStream` Class

This helper class handles webcam streaming in a separate thread to ensure smooth video capture. It continuously reads frames in the background, allowing face detection and recognition processes to run in parallel without blocking the video stream.

### 6. Real-time Processing and Parallelism

To ensure that the system can handle real-time video input efficiently, the webcam stream is processed in parallel using the `WebcamStream` class. This separates the video capture process from face detection and recognition, ensuring smooth performance even when multiple faces are detected per second.

## Getting Started

### Requirements

- Python 3.x
- OpenCV
- Dlib
- NumPy
- SQLite
- Tkinter (for the GUI)
- Pre-trained Dlib models for face landmarks and face recognition.

### Running the Project

1. Clone this repository.
2. Download the required models and place them in the `weights` directory:
   - **Face landmarks model**: `shape_predictor_68_face_landmarks.dat`
   - **Face recognition model**: `dlib_face_recognition_resnet_model_v1.dat`
3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the main application:
   ```bash
   python main.py
   ```

## How it Works

- The application starts the webcam stream and processes frames every N-th frame to optimize performance.
- For each frame, faces are detected, and their embeddings are generated.
- The embeddings are compared against the existing entries in the database to determine if the face already exists (based on a threshold tolerance of 0.6).
- If the face does not exist in the database, it is stored with the name "Unknown."
- The GUI allows users to label unknown faces by updating their names in the database.


## License

This project is open-source and available under the MIT License.
