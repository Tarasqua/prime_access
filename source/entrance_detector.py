"""
@tarasqua
"""
import datetime
import os
import shutil
from pathlib import Path
import asyncio
from itertools import chain
from collections import deque, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon

from background_subtractor import BackgroundSubtractor
from utils.math_funtions import *


class EntranceDetector:
    """Детектор входа и выхода людей в секторе"""

    def __init__(self, frame_shape: np.array, frame_dtype, roi: np.array,
                 yolo_model: str = 'n', yolo_confidence: float = 0.25,
                 fg_history: int = 1000, fg_threshold: int = 16, fg_detect_shadows: bool = True):
        area_threshold = Polygon(roi).area * 0.2  # движущийся объект должен занимать часть ROI
        self.bg_subtractor = BackgroundSubtractor(
            frame_shape, area_threshold, fg_history, fg_threshold, fg_detect_shadows)

        # маска roi для вычитания фона и трекинга (раздуваем полигон для трекинга)
        self.bg_stencil = self.__get_roi_mask(frame_shape, frame_dtype, roi)
        self.det_stencil = self.__get_roi_mask(  # раздуваем маску для лучшей работы YOLO
            frame_shape, frame_dtype, self.__inflate_polygon(roi, 1.75).astype(int))

        self.yolo_pose = self.__set_yolo_model(yolo_model)
        self.yolo_confidence = yolo_confidence
        # чтобы не было провисания во время рантайма
        self.yolo_pose.predict(
            np.random.randint(255, size=(300, 300, 3), dtype=np.uint8), classes=[0], verbose=False)

        self.tracking_people = {}
        self.collected_data = {}
        self.current_frame = None

        self.take_picture_counter = 50
        self.actions_threshold = 30
        self.moving_average_window = 10

    @staticmethod
    def __set_yolo_model(yolo_model) -> YOLO:
        """
        Выполняет проверку на то, загружена ли модель и, если нет, подгружает и возвращает
        Parameters:
            yolo_model: n (nano), m (medium)...
        Returns:
            YOLO модель
        """
        model = os.path.join(
            Path(__file__).resolve().parents[1], 'models', 'yolo_models', f'yolov8{yolo_model}-pose.onnx')
        if not os.path.exists(model):
            yolo_pose = YOLO(f'yolov8{yolo_model}-pose').export(format='onnx')
            shutil.move(f'yolov8{yolo_model}-pose.onnx', model)
            os.remove(f'yolov8{yolo_model}-pose.pt')
        else:
            yolo_pose = YOLO(model)
        return yolo_pose

    @staticmethod
    def __inflate_polygon(polygon_points: np.array, scale_multiplier: float) -> np.array:
        """
        Раздувает полигон точек
        Parameters:
            polygon_points: полигон точек вида np.array([[x, y], [x, y], ...])
            scale_multiplier: во сколько раз раздуть рамку
        Returns:
            inflated_polygon: раздутый полигон того же вида, что и входной
        """
        centroid = Polygon(polygon_points).centroid
        inflated_polygon = []
        for point in polygon_points:
            rho, phi = cart2pol(point[0] - centroid.x, point[1] - centroid.y)
            x, y = pol2cart(rho * scale_multiplier, phi)
            inflated_polygon.append([x + centroid.x, y + centroid.y])
        return np.array(inflated_polygon)

    @staticmethod
    def __get_roi_mask(frame_shape: tuple[int, int, int], frame_dtype: np.dtype, roi: np.array) -> np.array:
        """
        Создает маску, залитую черным вне ROI
        Parameters:
            frame_shape: размер изображения (height, width, channels)
            frame_dtype: dtype изображения
            roi: набор точек roi вида np.array([[x, y], [x, y], ...])
        Returns:
            stencil: черно-белая маска
        """
        stencil = np.zeros(frame_shape).astype(frame_dtype)
        cv2.fillPoly(stencil, [roi], (255, 255, 255))
        return stencil

    async def __update_track(self, detection) -> None:
        """
        Обработка затреченных людей в ROI:
            Записывает id трека.
            Каждые N кадров записывает обрезанный кадр с человеком.
            Определяет, входит человек (True) или выходит (False), сравнивая координаты по Х левого и правого плеча,
            исходя из того, что дверь справа.
            А также координаты центроида для определения - стоял человек на месте или шел.
        Parameters:
            detection: результат работы YOLO-детектора по одному (!) человеку
        """
        human_id = detection.boxes.id.numpy()[0].astype(int)
        is_entering = (detection.keypoints.data.numpy()[0][5][0] >  # левое плечо
                       detection.keypoints.data.numpy()[0][6][0])  # правое плечо
        # Счетчик кадров наблюдения; кадры детекции; координаты центроида; статистика вошел/вышел
        self.tracking_people.setdefault(human_id, [1, deque(maxlen=25), deque(maxlen=60), deque(maxlen=60)])
        self.tracking_people[human_id][0] += 1  # счетчик трека
        x1, y1, x2, y2 = detection.boxes.data.numpy()[0][:4].astype(int)
        if self.tracking_people[human_id][0] % self.take_picture_counter:
            self.tracking_people[human_id][1].append(self.current_frame[y1:y2, x1:x2])  # пикчи
        self.tracking_people[human_id][2].append([x1 + x2 / 2, y1 + y2 / 2])  # центроид
        self.tracking_people[human_id][3].append(is_entering)  # вошел / вышел

    @staticmethod
    def __was_track_moving(centroids: np.array) -> bool:
        """
        Определяет, шел человек или стоял на месте в ROI:
            С помощью polyfit смотрим, какой угол наклона у линейной функции, полученной из координат центроида.
            Если больше нуля, то человек двигался, иначе - стоял на месте.
        Parameters:
            centroids: массив центроидов по человеку в формате np.array([[x, y], [x, y], ...])
        Returns:
            was_moving: True, если да, False - нет
        """
        k, _ = np.polyfit(centroids[:, 0], centroids[:, 1], 1)
        return 25 < np.abs(np.degrees(np.arctan(k))) < 90  # смотрим, чтобы угол был острый

    async def collect_data(self, human_id: int, tracking_data: list) -> None:
        """
        Запись полученных данных по треку (одному) в итоговый словарь для отправки на сервер для
        дальнейшей детекции идентификации
            Формат выходных данных:
            id + 1 = tuple([кадры с людьми], вошел/вышел, время и дата обнаружения в формате datetime)
        Parameters:
            human_id: id, который нужно присвоить полученным данным
            tracking_data: данные по треку - счетчик кадров наблюдения; кадры детекции;
                        координаты центроида; статистика вошел/вышел
        """
        frames_counter, bboxes, centroids, is_entering = tracking_data
        if frames_counter > self.actions_threshold:
            if self.__was_track_moving(np.array(centroids)):
                self.collected_data[human_id] = \
                    (bboxes, is_entering.count(True) > is_entering.count(False), datetime.datetime.now())
        print(human_id)

    async def detect_(self, current_frame: np.ndarray) -> None:
        """
        Обработка кадра детектором контроля доступа:
            - получение маски движения в ROI;
            - если движение есть, включается YOLO-трекер;
            - если в раздутой маске ROI (для большей точности работы YOLO) обнаружены люди, они заносятся в
                отдельный временный словарь с текущими треками;
            - как только движения прекратились (человек/люди зашли/вышли), данные обрабатываются и закидываются
                на сервер для дальнейшей детекции лиц и хранения данных в базе.
        Формат выходных данных на сервер:
            dict: keys - id, values - tuple([кадры с людьми], вошел/вышел, время и дата обнаружения в формате datetime)
        Parameters:
            current_frame: текущий кадр, пришедший со стрима
        """
        self.current_frame = current_frame  # чтобы вырезать человека не из маски, а из ориг. изображения
        fg_bboxes = self.bg_subtractor.get_fg_bboxes(cv2.bitwise_and(self.current_frame, self.bg_stencil))
        if fg_bboxes.size != 0:
            # если есть движение в roi, начинаем тречить людей
            detections = self.yolo_pose.track(
                cv2.bitwise_and(self.current_frame, self.det_stencil),  # трек в раздутой roi
                classes=[0], verbose=False, persist=True, conf=self.yolo_confidence)[0]
            if detections.boxes.data.numpy().shape[0] != 0:  # если нашли кого-либо
                tracking_tasks = [asyncio.create_task(self.__update_track(detection))  # добавляем в треки
                                  for detection in detections if detection.boxes.id is not None]  # отсекаем без id
                [await task for task in tracking_tasks]
        else:
            # Перестраховываемся, чтобы даже если YOLO ре-идентифицировал человека и присвоил ему тот же id
            # мы все равно записали его несколько раз - при выходе и скором входе. А так как id нужен лишь для того,
            # чтобы данные не перемешивались, то даже если они будут пропускаться - не важно. Главное - уникальность
            last_collected_id = max(self.collected_data.keys()) if self.collected_data else 0
            # обрабатываем полученную информацию по трекам
            collecting_tasks = [asyncio.create_task(self.collect_data(last_collected_id + human_id, tracking_data))
                                for human_id, tracking_data in self.tracking_people.items()]
            [await task for task in collecting_tasks]
            self.tracking_people = {}  # обнуляем треки
