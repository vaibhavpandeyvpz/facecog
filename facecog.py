import cv2
import dlib
import json
import msgpack
import msgpack_numpy as m
import numpy
from os import path
import os
from PIL import ImageFont, ImageDraw, Image

m.patch()

CWD = path.dirname(__file__)
RESIZE = 4

# https://github.com/davisking/dlib-models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path.join(CWD, 'models', 'shape_predictor_5_face_landmarks.dat'))
recognition = dlib.face_recognition_model_v1(path.join(CWD, 'models', 'dlib_face_recognition_resnet_model_v1.dat'))


def detect_faces(image):
    bounds = []
    fds = []
    faces = detector(image, 1)
    for _, face in enumerate(faces):
        shape = predictor(image, face)
        fd = recognition.compute_face_descriptor(image, shape, 1)
        bounds.append((face.top(), face.right(), face.bottom(), face.left()))
        fds.append(numpy.array(fd))
    return bounds, fds


def find_abouts():
    persons = path.join(CWD, 'persons')
    abouts = {}
    for _, folder in enumerate(os.listdir(persons)):
        about = path.join(persons, folder, 'about.json')
        with open(about) as f:
            abouts[folder] = json.load(f)
    return abouts


def find_persons():
    ids = []
    faces = []
    persons = path.join(CWD, 'persons')
    for _, folder in enumerate(os.listdir(persons)):
        photos = path.join(persons, folder, 'photos')
        for _, photo in enumerate(os.listdir(photos)):
            print('Processing %s for %s ...' % (photo, folder))
            image = load_image(path.join(photos, photo))
            _, found = detect_faces(image)
            faces.extend(found)
            for _ in range(0, len(found)):
                ids.append(folder)
    return ids, faces


def load_image(file):
    bgr = cv2.imread(file)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def run_app():
    abouts = find_abouts()
    cache = path.join(CWD, 'cache', 'persons.packed')
    if path.exists(cache):
        with open(cache, 'rb') as f:
            ids, known = msgpack.loads(f.read())
    else:
        ids, known = find_persons()
        with open(cache, 'wb') as f:
            f.write(msgpack.dumps((ids, known)))
    roboto = ImageFont.truetype(path.join(CWD, 'fonts', 'Roboto', 'Roboto-Regular.ttf'), 12)
    print('Know about %d persons with %d samples.' % (len(abouts), len(known)))
    cv2.namedWindow('Preview', cv2.WINDOW_AUTOSIZE)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    skip = False
    matches = []
    while True:
        _, frame = cap.read()
        if not skip:
            matches = []
            mini = cv2.resize(frame, (0, 0), fx=1 / RESIZE, fy=1 / RESIZE)
            rgb = cv2.cvtColor(mini, cv2.COLOR_BGR2RGB)
            rects, found = detect_faces(rgb)
            for i, face in enumerate(found):
                matched = numpy.linalg.norm(known - face, axis=1)
                for j, difference in enumerate(matched):
                    if difference <= 0.5:
                        matches.append((ids[j], difference, rects[i]))
        skip = not skip
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for _, match in enumerate(matches):
            about = abouts[match[0]]
            print('%s matches with %s tolerance.' % (about['name'], match[1]))
            top, right, bottom, left = (match[2][0] * RESIZE,
                                        match[2][1] * RESIZE,
                                        match[2][2] * RESIZE,
                                        match[2][3] * RESIZE)
            draw.rectangle([(left, top), (right, bottom)], outline='red')
            draw.rectangle([(left, top - 3 - 15 - 3 - 15 - 3), (right, top)], fill='red')
            draw.text((left + 3, top - 3 - 15 - 15), about['name'], fill='white', font=roboto)  # occupation
            draw.text((left + 3, top - 3 - 15), about['occupation'], fill='white', font=roboto)  # name
        bgr = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
        cv2.imshow('Preview', bgr)
        if cv2.waitKey(1) == 27:  # [Esc]
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_app()
