"""
...
"""
import datetime
import os
import shutil
from pathlib import Path
import asyncio
from itertools import chain

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon

from background_subtractor import BackgroundSubtractor
from roi_polygon import ROIPolygon
from plots import Plots


class EntranceDetector:
    """..."""

    def __init__(self, frame_shape: np.array, frame_dtype, roi: np.array, plot_detections: bool = False,
                 yolo_model: str = 'n', yolo_confidence: float = 0.7,
                 fg_history: int = 500, fg_threshold: int = 16, fg_detect_shadows: bool = True):
        area_threshold = Polygon(roi).area * 0.5
        self.bg_subtractor = BackgroundSubtractor(
            frame_shape, area_threshold, fg_history, fg_threshold, fg_detect_shadows)

        # маска roi для вычитания фона и трекинга (раздуваем полигон для трекинга)
        self.bg_stencil = np.zeros(frame_shape).astype(frame_dtype)
        cv2.fillPoly(self.bg_stencil, [roi], (255, 255, 255))
        self.det_stencil = np.zeros(frame_shape).astype(frame_dtype)
        cv2.fillPoly(self.det_stencil, [self.inflate_polygon(roi, 1.35).astype(int)], (255, 255, 255))

        self.yolo_pose = self.__set_yolo_model(yolo_model)
        self.yolo_confidence = yolo_confidence
        self.yolo_pose.predict(cv2.imread('test.png'), classes=[0], verbose=False)

        self.plots = Plots(yolo_confidence)
        self.plot_detections = plot_detections

    @staticmethod
    def cart2pol(x, y):
        """Декартовы в полярные"""
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    @staticmethod
    def pol2cart(rho, phi):
        """Полярные в декартовы"""
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def inflate_polygon(self, polygon_points: np.array, scale_multiplier: float):
        """Раздувает полигон точек"""
        centroid = Polygon(polygon_points).centroid
        inflated_polygon = []
        for point in polygon_points:
            rho, phi = self.cart2pol(point[0] - centroid.x, point[1] - centroid.y)
            x, y = self.pol2cart(rho * scale_multiplier, phi)
            inflated_polygon.append([x + centroid.x, y + centroid.y])
        return np.array(inflated_polygon)

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

    async def detect_(self, current_frame: np.ndarray) -> np.array or None:
        """..."""
        fg_bboxes = self.bg_subtractor.get_fg_bboxes(cv2.bitwise_and(current_frame, self.bg_stencil))
        if fg_bboxes.size != 0:
            # detections = self.__detect_pose(cv2.bitwise_and(current_frame, self.det_stencil))
            detections = self.yolo_pose.track(
                cv2.bitwise_and(current_frame, self.det_stencil), classes=[0], verbose=False, persist=True)[0]
            if self.plot_detections and detections.boxes.data.numpy().shape[0] != 0:
                current_frame = await self.plots.plot_kpts(current_frame, detections)
        return current_frame if self.plot_detections else None
