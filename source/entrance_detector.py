"""
...
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
from plots import Plots
from utils.math_funtions import *


class EntranceDetector:
    """..."""

    def __init__(self, frame_shape: np.array, frame_dtype, roi: np.array, plot_detections: bool = False,
                 yolo_model: str = 'n', yolo_confidence: float = 0.25,
                 fg_history: int = 1000, fg_threshold: int = 16, fg_detect_shadows: bool = True):
        area_threshold = Polygon(roi).area * 0.2
        self.bg_subtractor = BackgroundSubtractor(
            frame_shape, area_threshold, fg_history, fg_threshold, fg_detect_shadows)

        # маска roi для вычитания фона и трекинга (раздуваем полигон для трекинга)
        self.bg_stencil = self.__get_roi_mask(
            frame_shape, frame_dtype, roi)
        self.det_stencil = self.__get_roi_mask(
            frame_shape, frame_dtype, self.__inflate_polygon(roi, 1.75).astype(int))

        self.yolo_pose = self.__set_yolo_model(yolo_model)
        self.yolo_confidence = yolo_confidence
        self.yolo_pose.predict(cv2.imread('test.png'), classes=[0], verbose=False)

        self.plots = Plots(yolo_confidence)
        self.plot_detections = plot_detections

        self.detected_people = {}
        self.collected_data = {}

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

    async def update_detection(self, detection):
        """..."""
        human_id = detection.boxes.id.numpy()[0].astype(int)
        is_entering = detection.keypoints.data.numpy()[0][5][0] > detection.keypoints.data.numpy()[0][6][0]
        # Счетчик кадров наблюдения; кадры детекции; статистика вошел/вышел
        self.detected_people.setdefault(human_id, [1, deque(maxlen=20), deque(maxlen=60)])
        self.detected_people[human_id][0] += 1
        if self.detected_people[human_id][0] % 25:
            x1, y1, x2, y2 = detection.boxes.data.numpy()[0][:4].astype(int)
            self.detected_people[human_id][1].append(detection.orig_img[y1:y2, x1:x2])
        self.detected_people[human_id][2].append(is_entering)

    async def collect_data(self):
        """..."""
        for (frames_counter, bboxes, is_entering) in self.detected_people.values():
            if frames_counter < 20:  # отсекаем мелкие движения и потери трека
                continue
            new_id = (max(self.collected_data.keys()) + 1) if self.collected_data else 1
            # закидываем по новому id ббоксы и зашел или вышел человек
            self.collected_data[new_id] = (bboxes, is_entering.count(True) > is_entering.count(False))
        self.detected_people = {}  # обнуляем треки

    async def detect_(self, current_frame: np.ndarray) -> np.array or None:
        """..."""
        fg_bboxes = self.bg_subtractor.get_fg_bboxes(cv2.bitwise_and(current_frame, self.bg_stencil))
        if fg_bboxes.size != 0:
            # если есть движение в roi, начинаем тречить людей
            detections = self.yolo_pose.track(
                cv2.bitwise_and(current_frame, self.det_stencil),  # трек в раздутой roi
                classes=[0], verbose=False, persist=True, conf=self.yolo_confidence)[0]
            if detections.boxes.data.numpy().shape[0] != 0:  # если нашли кого-либо
                tasks = [asyncio.create_task(self.update_detection(detection)) for detection in detections]
                [await task for task in tasks]
                # cv2.circle(current_frame, detections.keypoints.data.numpy()[0][5][:-1].astype(int), 5,
                #            (0, 255, 0), -1)
                # cv2.circle(current_frame, detections.keypoints.data.numpy()[0][6][:-1].astype(int), 5,
                #            (0, 0, 255), -1)
        else:
            await self.collect_data()
            # if self.plot_detections and detections.boxes.data.numpy().shape[0] != 0:
            #     current_frame = await self.plots.plot_kpts(current_frame, detections)
        return current_frame if self.plot_detections else None
