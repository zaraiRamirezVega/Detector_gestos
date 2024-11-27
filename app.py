
# -*- coding: utf-8 -*-
import argparse
import copy
import csv
import itertools
from collections import Counter, deque

import cv2 as cv
import mediapipe as mp
import numpy as np

from model import KeyPointClassifier, PointHistoryClassifier
from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Análisis de argumentos (parsing de argumentos)
    args = get_args()

    # Configuración de la cámara
    cap_device = args.device  # Dispositivo de la cámara (por ejemplo, 0 para la cámara predeterminada)
    cap_width = args.width    # Ancho del video
    cap_height = args.height  # Alto del video

    # Configuración de la detección y seguimiento de manos
    use_static_image_mode = args.use_static_image_mode  # Modo de imagen estática (si se usa imagen estática o video)
    min_detection_confidence = args.min_detection_confidence  # Confianza mínima para detección de manos
    min_tracking_confidence = args.min_tracking_confidence  # Confianza mínima para seguimiento de manos

    # Activación del uso de rectángulos para mostrar los límites de las manos
    use_brect = True

    # Preparación de la cámara
    cap = cv.VideoCapture(cap_device)  # Se abre la cámara
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)  # Establecer el ancho de la imagen de la cámara
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)  # Establecer la altura de la imagen de la cámara

    # Carga del modelo de detección de manos
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,  # Modo de imagen estática o no
        max_num_hands=2,  # Número máximo de manos que se detectarán
        min_detection_confidence=min_detection_confidence,  # Confianza mínima para la detección
        min_tracking_confidence=min_tracking_confidence,  # Confianza mínima para el seguimiento
    )

    # Cargamos los clasificadores para las posiciones de los puntos clave y el historial de puntos
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Lectura de las etiquetas para el clasificador de puntos clave
    with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]  # Guardamos las etiquetas

    # Lectura de las etiquetas para el clasificador del historial de puntos
    with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]  # Guardamos las etiquetas

    # Medición de FPS (cuadros por segundo) para mostrar la velocidad del procesamiento
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Configuración del historial de coordenadas (para gestos de mano)
    history_length = 16  # Longitud del historial de puntos
    point_history = deque(maxlen=history_length)  # Cola que almacena el historial de puntos

    # Historia de los gestos de los dedos
    finger_gesture_history = deque(maxlen=history_length)  # Cola para el historial de gestos

    # Modo inicial
    mode = 0

    while True:
        fps = cvFpsCalc.get()  # Obtener FPS actual

        # Procesar la tecla presionada (ESC: finalizar)
        key = cv.waitKey(10)
        if key == 32:  # ESPACIO
            break
        number, mode = select_mode(key, mode)  # Cambiar el modo según la tecla presionada

        # Captura de imagen desde la cámara
        ret, image = cap.read()
        if not ret:
            break  # Si no se puede leer la imagen, se sale del ciclo
        image = cv.flip(image, 1)  # Reflejo horizontal de la imagen (para simular espejo)
        debug_image = copy.deepcopy(image)  # Copia de la imagen para fines de depuración

        # Implementación de la detección de manos
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convertir la imagen a RGB para procesar con MediaPipe

        image.flags.writeable = False  # Desactivar la escritura en la imagen original
        results = hands.process(image)  # Procesar la imagen para detectar manos
        image.flags.writeable = True  # Volver a habilitar la escritura en la imagen

        if results.multi_hand_landmarks is not None:  # Si se detectaron manos
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Cálculo del cuadro delimitador (rectángulo) de la mano
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Cálculo de los puntos clave de la mano (landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversión a coordenadas relativas (normalizadas)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                # Guardar los datos en un archivo CSV para análisis posterior
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                # Clasificación de signos de la mano
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Si el gesto es el dedo apuntando (gesto de punto)
                    point_history.append(landmark_list[8])  # Agregar punto al historial
                else:
                    point_history.append([0, 0])  # No es un gesto de punto

                # Clasificación de gestos de los dedos
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):  # Si hay suficiente historial de puntos
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                # Calculamos los gestos más comunes en la última detección
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                # Parte de dibujo: mostrar los resultados en la imagen
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])  # Si no se detectaron manos, agregar puntos vacíos

        # Dibujo del historial de puntos
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Mostrar la imagen con las predicciones y los resultados
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()  # Liberar la cámara
    cv.destroyAllWindows()  # Cerrar las ventanas de OpenCV



def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Pulgar
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 182, 193), 6)  # Rosa pastel
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 105, 180), 2)  # Rosa fuerte
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 182, 193), 6)  # Rosa pastel
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 105, 180), 2)  # Rosa fuerte

        # Dedo índice
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (173, 216, 230), 6)  # Azul claro
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (135, 206, 250), 2)  # Azul cielo
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (173, 216, 230), 6)  # Azul claro
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (135, 206, 250), 2)  # Azul cielo
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (173, 216, 230), 6)  # Azul claro
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (135, 206, 250), 2)  # Azul cielo

        # Dedo medio
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 228, 225), 6)  # Rosa claro
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 160, 122), 2)  # Rosa coral
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 228, 225), 6)  # Rosa claro
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 160, 122), 2)  # Rosa coral
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 228, 225), 6)  # Rosa claro
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 160, 122), 2)  # Rosa coral

        # Dedo anular
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 218, 185), 6)  # Durazno claro
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 99, 71), 2)  # Rojo tomate
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 218, 185), 6)  # Durazno claro
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 99, 71), 2)  # Rojo tomate
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 218, 185), 6)  # Durazno claro
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 99, 71), 2)  # Rojo tomate

        # Dedo meñique
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 222, 173), 6)  # Amarillo claro
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 215, 0), 2)  # Amarillo brillante
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 222, 173), 6)  # Amarillo claro
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 215, 0), 2)  # Amarillo brillante
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 222, 173), 6)  # Amarillo claro
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 215, 0), 2)  # Amarillo brillante

        # Palma
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 240, 245), 6)  # Rosa muy claro
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 105, 180), 2)  # Rosa fuerte
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 240, 245), 6)  # Rosa muy claro
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 105, 180), 2)  # Rosa fuerte
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 240, 245), 6)  # Rosa muy claro
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 105, 180), 2)  # Rosa fuerte
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 240, 245), 6)  # Rosa muy claro
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 105, 180), 2)  # Rosa fuerte
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 240, 245), 6)  # Rosa muy claro
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 105, 180), 2)  # Rosa fuerte
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 240, 245), 6)  # Rosa muy claro
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 105, 180), 2)  # Rosa fuerte
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 240, 245), 6)  # Rosa muy claro
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 105, 180), 2)  # Rosa fuerte


    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15: 
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16: 
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Gestos con los dedos:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Gestos con los dedos:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Punto clave registro', 'Historial punto de registro']
    if 1 <= mode <= 2:
        cv.putText(image, "MODO:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
