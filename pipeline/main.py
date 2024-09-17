import dlib
import cv2
from collections import namedtuple
import numpy as np
import sqlite3
import os
import tkinter as tk
from tkinter import Label, Button, Entry, filedialog
from PIL import Image, ImageTk

# code source: https://openface-api.readthedocs.io/en/latest/_modules/openface/align_dlib.html#AlignDlib
# Create class AlignFace to detect landmarks + centered the face
TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])
TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
OUTER_EYES_AND_NOSE = [36, 45, 33]

FaceInformations = namedtuple("FaceInformations", ("name", "face", "face_embd"))

class FaceEmbeddings:
    def __init__(self, landmarks_path_model, face_recognition_path_model):
        self.pose_68_point_model = dlib.shape_predictor(landmarks_path_model)
        self.face_encoder = dlib.face_recognition_model_v1(face_recognition_path_model)
        self.face_detector = dlib.get_frontal_face_detector()
    
    def import_image(self, image_path, resize_img=None):
        image = cv2.imread(image_path)
        if resize_img is not None:
            image = cv2.resize(image, resize_img)
        return image
    
    def detect_face(self, image):
        return self.face_detector(image, 1)
    
    def landmarks_face(self, image, bound_box):
        return self.pose_68_point_model(image, bound_box)
    
    def face_encodings(self, image, landmarks_points, num_jitters=1):
        return self.face_encoder.compute_face_descriptor(image, landmarks_points, num_jitters)

    def align_face(self, image, img_dim, bound_box):
        lms_points = self.landmarks_face(image, bound_box)
        landmarks = np.float32(list(map(lambda s: (s.x, s.y), lms_points.parts())))
        landmarks_indices = np.array(OUTER_EYES_AND_NOSE) 
        H = cv2.getAffineTransform(landmarks[landmarks_indices],
                                img_dim * MINMAX_TEMPLATE[landmarks_indices])
        thumbnail = cv2.warpAffine(image, H, (img_dim, img_dim))
        return thumbnail
    
    def pipeline(self, image_path):
        image = self.import_image(image_path)
        detected_faces = self.detect_face(image)
        face_info_list = []
        for i, det_face in enumerate(detected_faces):
            landmarks_face = self.pose_68_point_model(image, det_face)
            face_embd = self.face_encodings(image, landmarks_face)
            face = self.align_face(image, 530, det_face)
            face_info = FaceInformations("Unknown", face, face_embd)
            face_info_list.append(face_info)
        return face_info_list
    

# Checking if the database exists and deleting data if it does
def check_and_delete_database(dataset_manager):
    if os.path.exists(dataset_manager.db_path):
        print(f"The database '{dataset_manager.db_path}' exists.")
        dataset_manager.delete()
    else:
        print(f"The database '{dataset_manager.db_path}' does not exist.")


class DatasetManagerSQLite:    
    def __init__(self, db_path="face_data.db"):
        self.db_path = db_path
        self.con = sqlite3.connect(db_path)
        self.create_table()
    
    def create_table(self):
        """ Create table to store face data if it doesn not exist. """
        query = '''
        CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        embeddings TEXT,
        face_image BLOB
        )
        '''
        self.con.execute(query)
        self.con.commit()
    
    def save_face_info(self, face_info_list):
        """ Save face information to the SQLite database. """
        query = 'INSERT INTO faces (name, embeddings, face_image) VALUES (?,?,?)'
        for face_info in face_info_list:
            embeddings_str = np.array2string(np.array(face_info.face_embd), separator=",")
            # face_image_str = np.array2string(np.array(face_info.face.flatten()), separator=",")
            _, img_encoded = cv2.imencode(".png", face_info.face)
            img_binary = img_encoded.tobytes()
            self.con.execute(query, (face_info.name, embeddings_str, img_binary))
        self.con.commit()
    
    def delete(self):
        """ Delete all rows in the faces table. """
        query = 'DELETE FROM faces'
        self.con.execute(query)
        self.con.commit()
        print("All rows have been deleted from the 'faces' table.")
    
    def update_face_name(self, new_name, id):
        """ Update the name of a person in the database """
        query = '''UPDATE faces SET name = ? WHERE id = ?'''
        self.con.execute(query, (new_name, id))
        self.con.commit()

class FaceDatabaseGUI:
    def __init__(self, root, dataset_manager):
        self.root = root
        self.dataset_manager = dataset_manager
        self.face_data = []
        self.current_index = 0
        
        self.root.title("Face Database Manager")
        
        # Create UI components
        self.name_label = Label(root, text="Name:")
        self.name_label.grid(row=0, column=0)
        
        self.name_entry = Entry(root)
        self.name_entry.grid(row=0, column=1)
        
        self.face_label = Label(root)
        self.face_label.grid(row=1, column=0, columnspan=2)
        
        self.prev_button = Button(root, text="Previous", command=self.show_previous_face)
        self.prev_button.grid(row=2, column=0)
        
        self.next_button = Button(root, text="Next", command=self.show_next_face)
        self.next_button.grid(row=2, column=1)
        
        self.update_button = Button(root, text="Update Name", command=self.update_name)
        self.update_button.grid(row=3, column=0, columnspan=2)
        
        # Load face data from the database
        self.load_face_data()
        if self.face_data:
            self.display_face(self.current_index)

    def load_face_data(self):
        """ Retrieve all faces from the database, including the id """
        query = 'SELECT id, name, face_image FROM faces'
        cursor = self.dataset_manager.con.execute(query)
        self.face_data = cursor.fetchall()

    def display_face(self, index):
        """ Display the face image and name at the given index """
        if not self.face_data:
            return
        
        face_id, name, face_image = self.face_data[index]
        self.name_entry.delete(0, tk.END)
        self.name_entry.insert(0, name)
        
        # Convert the binary image back to a format Tkinter can display
        img_array = np.frombuffer(face_image, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        self.face_label.configure(image=img_tk)
        self.face_label.image = img_tk  # Keep a reference to avoid garbage collection

    def show_previous_face(self):
        """ Navigate to the previous face """
        if self.current_index > 0:
            self.current_index -= 1
            self.display_face(self.current_index)

    def show_next_face(self):
        """ Navigate to the next face """
        if self.current_index < len(self.face_data) - 1:
            self.current_index += 1
            self.display_face(self.current_index)

    def update_name(self):
        """ Update the name of the current face in the database using the id """
        new_name = self.name_entry.get()
        face_id, _, _ = self.face_data[self.current_index]
        
        # Update in the database using the face_id
        self.dataset_manager.update_face_name(new_name, face_id)
        print(f"Updated name to {new_name}")

        # Reload data to ensure changes are reflected
        self.load_face_data()


if __name__ == "__main__":

    # Image path & weight parameter tuned path
    img_path = "rhcp.jpg"
    landmarks_path_model = "weights/WEIGHTS_face_landmarks/shape_predictor_68_face_landmarks.dat"
    face_recognition_path_model = "weights/WEIGHTS_deepCNN_face_recognition/dlib_face_recognition_resnet_model_v1.dat"

    # Process the image and create the embeddings of each face
    face_embeddings = FaceEmbeddings(landmarks_path_model, face_recognition_path_model)
    face_info_list = face_embeddings.pipeline(img_path)
    
    # Initialize the dataset and save the initial face info list
    dataset_manager = DatasetManagerSQLite("face_data.db")
    check_and_delete_database(dataset_manager)
    dataset_manager.save_face_info(face_info_list)

    # Set up the Tkinter root window
    root = tk.Tk()
    gui = FaceDatabaseGUI(root, dataset_manager)
    root.mainloop()

