import cv2
import dlib
import re
import os
import numpy as np

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    # Loop in each image of the folder
    for i, file in enumerate(image_files_in_folder(known_people_folder)[:1]):
        basename = os.path.splitext(os.path.basename(file))[0]
        print(basename)
        img = cv2.imread(file) # np.array type


def scan_known_people(known_people_folder):
    known_names = os.listdir(known_people_folder)

    for name in known_names:
        dir_known_name = os.path.join(known_people_folder, name)
        for i, file in enumerate(image_files_in_folder(dir_known_name)):
            print(file)



scan_known_people("training-images")


