import streamlit as st
import cv2
import numpy as np
import os
import pickle
from PIL import Image
from numpy import asarray, expand_dims
from yoloface import face_analysis
import mediapipe as mp
from keras.models import load_model

# Initialize YOLO Face
faceY = face_analysis()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def load_facenet_model():
    model = None
    try:
        from keras_facenet import FaceNet
        model = FaceNet()
    except Exception as e:
        st.error(f"Failed to initialize FaceNet with keras_facenet: {e}")
        try:
            model = load_model('path/to/facenet_model.h5')
        except Exception as e:
            st.error(f"Failed to load pre-trained FaceNet model: {e}")
    return model

MyFaceNet = load_facenet_model()

def register_student(name, roll):
    cap = cv2.VideoCapture(0)
    progress_text = st.empty()
    startX, startY, endX, endY = 0, 0, 0, 0
    dataset_path = 'training_data10/'
    assure_path_exists(dataset_path)
    count = 0
    
    st.write("Capturing images, please look straight at the camera and tilt your head left and right...")
    
    while count < 10:
        ret, frame = cap.read()
        _, box, conf = faceY.face_detection(frame_arr=frame, frame_status=True, model='full')

        for i, rbox in enumerate(box):
            if conf[i] > 0.5:
                startX = rbox[0]
                startY = rbox[1]
                endX = rbox[0] + rbox[3]
                endY = rbox[1] + rbox[2]

        img_h, img_w, _ = frame.shape

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                face_3d = []
                face_2d = []

                for idx, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if not frame.any():
                    st.error("Error: Empty frame. Unable to capture the image.")
                    break
                count += 1

                if startY < endY and startX < endX and 0 <= startY < frame.shape[0] and 0 <= endY <= frame.shape[0] and 0 <= startX < frame.shape[1] and 0 <= endX <= frame.shape[1]:
                    if y < -10:
                        cv2.imwrite(f"{dataset_path}{name}p.{roll}.{count}.jpg", frame[startY:endY, startX:endX])
                    elif y > 10:
                        cv2.imwrite(f"{dataset_path}{name}p.{roll}.{count}.jpg", frame[startY:endY, startX:endX])
                    else:
                        cv2.imwrite(f"{dataset_path}{name}f.{roll}.{count}.jpg", frame[startY:endY, startX:endX])
                else:
                    st.error("Error: Invalid ROI. Unable to capture the image.")
                    count -= 1

            progress_text.text(f"Images captured: {count}/10")

    cap.release()
    cv2.destroyAllWindows()

    st.success("Dataset captured successfully!")

def generate_embeddings():
    folder = 'training_data10/'
    database = {}

    for filename in os.listdir(folder):
        path = folder + filename
        gbr1 = cv2.imread(path)

        gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
        gbr = Image.fromarray(gbr)
        gbr_array = np.asarray(gbr)
        face = Image.fromarray(gbr_array)
        face = face.resize((160, 160))
        face = np.asarray(face)

        face = np.expand_dims(face, axis=0)
        signature = MyFaceNet.embeddings(face)

        database[os.path.splitext(filename)[0].split(".")[0]] = signature

    with open("data.pkl", "wb") as myfile:
        pickle.dump(database, myfile)

    st.success("Embeddings generated and saved successfully!")

def grading():
    with open("data.pkl", "rb") as myfile:
        database = pickle.load(myfile)
    
    cap = cv2.VideoCapture(0)
    
    # Dictionary to store grades for each student
    Real_Grade = {}
    identity_dict = {}

    st.write("Starting attendance grading...")
    
    # Create Streamlit elements for displaying student grades
    grades_table = st.empty()

    while True:
        ret, frame = cap.read()
        _, box, conf = faceY.face_detection(frame_arr=frame, frame_status=True, model='full')

        for i, rbox in enumerate(box):
            if conf[i] > 0.5:
                startX = rbox[0]
                startY = rbox[1]
                endX = rbox[0] + rbox[3]
                endY = rbox[1] + rbox[2]
                
                try:
                    gbr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    gbr = Image.fromarray(gbr)
                    gbr_array = asarray(gbr)

                    face = gbr_array[startY:endY, startX:endX]

                    face = Image.fromarray(face)
                    face = face.resize((160, 160))
                    face = asarray(face)

                    face = expand_dims(face, axis=0)
                    signature = MyFaceNet.embeddings(face)

                    min_dist = 100
                    identity = ' '
                    for key, value in database.items():
                        dist = np.linalg.norm(value - signature)

                        if dist < min_dist:
                            min_dist = dist
                            identity = key

                    if identity[-1] == 'f':
                        if identity[:-1] in Real_Grade:
                            Real_Grade[identity[:-1]] += 0.2
                        else:
                            Real_Grade[identity[:-1]] = 0.2
                    else:
                        if identity[:-1] in Real_Grade:
                            Real_Grade[identity[:-1]] -= 0.2
                        else:
                            Real_Grade[identity[:-1]] = 0.2

                    identity_dict[identity[:-1]] = Real_Grade[identity[:-1]]
                    
                    # Display all students and their grades
                    grades_table.table({
                        "Student": list(identity_dict.keys()),
                        "Grade": list(identity_dict.values())
                    })

                except ValueError:
                    continue

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


def register():
    st.title("Student Registration and Attendance System")

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Register", "Generate Embeddings", "Start Attendance"])

    if selection == "Home":
        st.write("Welcome to the Student Registration and Attendance System.")
        st.write("Use the sidebar to navigate between registration, generating embeddings, and starting attendance grading.")

    elif selection == "Register":
        st.header("Register a Student")
        name = st.text_input("Enter your name:")
        roll = st.text_input("Enter your roll number:")
        generate_embeddings_checkbox = st.checkbox("I agree to use my images for attendance detection and generate embeddings.")

        if st.button("Register"):
            st.write("Please look straight at the camera and tilt your head left and right for dataset capturing.")
            register_student(name, roll)

            if generate_embeddings_checkbox:
                generate_embeddings()

    elif selection == "Generate Embeddings":
        st.header("Generate Embeddings")
        if st.button("Generate"):
            generate_embeddings()

    elif selection == "Start Attendance":
        st.header("Start Attendance Grading")
        grading()

if __name__ == "__main__":
    register()
