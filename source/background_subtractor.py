"""
...
"""
import cv2
import numpy as np


class BackgroundSubtractor:
    """
    Нахождение маски движения
    """

    def __init__(self, frame_shape: np.array, area_threshold: float,
                 history: int = 1500, threshold: int = 16, detect_shadows: bool = True):
        self.history, self.threshold, self.detect_shadows = history, threshold, detect_shadows
        self.back_sub_model = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=threshold, detectShadows=detect_shadows)
        self.dilate_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        self.area_threshold = area_threshold
        self.frame_shape = frame_shape[:-1][::-1]
        self.resize_shape = (np.array(self.frame_shape) / 8).astype(int)

    def __get_fg_mask(self, current_frame: np.array) -> np.ndarray:
        """
        Расчет маски движения
        Parameters:
            current_frame: текущий кадр
        Returns:
            fg_mask: маска движения
        """
        resized_frame = cv2.resize(current_frame, self.resize_shape)  # для большей производительности уменьшаем
        fg_mask = self.back_sub_model.apply(resized_frame)  # маска переднего плана
        fg_mask[fg_mask == 127] = 0  # удаляем тени
        eroding = cv2.erode(fg_mask, self.dilate_erode_kernel, iterations=1)  # эрозия
        dilating = cv2.dilate(eroding, self.dilate_erode_kernel, iterations=1)  # дилатация
        closing = cv2.morphologyEx(  # морфологическое закрытие
            dilating, cv2.MORPH_CLOSE, self.morph_close_kernel, iterations=2)
        return cv2.resize(closing, self.frame_shape)  # в исходный размер

    def get_fg_bboxes(self, current_frame: np.array) -> np.ndarray:
        """
        Возвращает bbox'ы, полученные из контуров маски переднего плана, отфильтрованные по площади
        Parameters:
            current_frame: текущий кадр
        Returns:
            fg_bboxes: np.ndarray формата [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        """
        fg_mask = self.__get_fg_mask(current_frame)
        # cv2.imshow('dst', cv2.resize(fg_mask, (np.array(fg_mask.shape[::-1]) / 2).astype(int)))
        contours, _ = cv2.findContours(  # находим контуры
            image=fg_mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        fg_bboxes = np.fromiter(map(  # распаковка из xywh в xyxy и фильтрация по площади от мелких шумов
            lambda xywh: np.array([xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]),
            [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > self.area_threshold]
        ), dtype=np.dtype((int, 4)))
        return fg_bboxes
