import tempfile

import streamlit as st
from PIL import Image

from model import *

swinging_stages = ['P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'P8', 'P10']

st.image(Image.open('assets/logo.png'), width=100)
st.markdown('''
<style>
img {
    user-select: none;
    pointer-events: none;
}
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
''', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Выберите видео свинга", type='.mp4', accept_multiple_files=False)

if uploaded_file is not None:
    selected = True
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    vf = cv2.VideoCapture(tfile.name)

    st.write('Обработка ' + uploaded_file.name)
    pose_clf = get_pose_clf('assets/pose_detection_light.tflite')
    landmarker = get_landmarker(
        'assets/pose_landmarker_lite.task')
    y_pred, frames = pose_mean_algo(pose_clf, tfile.name)

    points1 = find_landmarks(landmarker, frames[y_pred[0]])
    try:
        points2 = find_landmarks(landmarker, frames[y_pred[3]])
    except:
        points2 = find_landmarks(landmarker, frames[-1])
    try:
        dist = head_movement(points1, points2)
        if dist > 70:
            st.write(f'Найдена ошибка: голова сдвинута на {dist:.2f} пикселей')
        else:
            st.write(f'Не обнаружено движений головы')
    except:
        points1 = find_landmarks(landmarker, frames[0])
        points2 = find_landmarks(landmarker, frames[len(frames) // 4])
        try:
            dist = head_movement(points1, points2)
            if dist > 70:
                st.write(f'Найдена ошибка: голова сдвинута на {dist:.2f} пикселей')
            else:
                st.write(f'Не обнаружено движений головы')
        except:
            st.write('В кадре нет человека')
    st.write('Ракурс:', angle(points1))
    st.write('Стадии свинга:')
    for i in range(8):
        st.write(f'{swinging_stages[i]}: {y_pred[i]}')
