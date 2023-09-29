"""
Created by
@tarasqua 26/09/2023
"""
import os
from collections import deque
from itertools import product
from pathlib import Path

import cv2
import numpy as np

from tracker import Tracker
from auxiliary_functions import get_iou


class FrameProcessing:
    """
    ...
    """

    def __init__(self, frame_shape: np.array):
        self.frame_shape = frame_shape[:-1][::-1]
        self.resize_shape: np.array = (np.array(frame_shape[:-1][::-1]) / 5).astype(int)

        self.back_sub_model = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.dilate_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.morph_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.min_area = 20000

        ssd_models_path = os.path.join(Path(__file__).resolve().parents[1], 'models', 'ssd_models')
        self.ssd_net = cv2.dnn.readNetFromCaffe(  # SSD
                os.path.join(ssd_models_path,  # берем файлы по расширению, чтобы не учитывать их название
                             [file for file in os.listdir(ssd_models_path) if file.endswith('.prototxt')][0]),
                os.path.join(ssd_models_path,
                             [file for file in os.listdir(ssd_models_path) if file.endswith('.caffemodel')][0])
        )
        self.tracker = Tracker(history_threshold=90, min_hits=3, iou_threshold=0.3)
        self.human_bbox_tracks = dict()

    def get_contours_fg_mask(self, original_frame: np.array) -> tuple:
        """Возвращает контуры из маски движения и маску переднего плана"""
        resized_frame = cv2.resize(original_frame, self.resize_shape)  # для большей производительности уменьшаем
        fg_mask = self.back_sub_model.apply(resized_frame)  # маска переднего плана
        fg_mask[fg_mask == 127] = 0  # удаляем тени
        eroding = cv2.erode(fg_mask, self.dilate_erode_kernel, iterations=1)  # эрозия
        dilating = cv2.dilate(eroding, self.dilate_erode_kernel, iterations=2)  # дилатация
        closing = cv2.morphologyEx(  # морфологическое закрытие
                dilating, cv2.MORPH_CLOSE, self.morph_close_kernel, iterations=1)
        closing_resized_back = cv2.resize(closing, self.frame_shape)  # в исходный размер
        contours, _ = cv2.findContours(  # находим контуры
                image=closing_resized_back, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        return contours, closing_resized_back

    def check_contours(self, contour_bboxes: np.array):
        """"""
        # Получаем ббоксы из истории
        last_hist_bboxes = [np.concatenate(bboxes[-1]) for bboxes in self.human_bbox_tracks.values()]
        if contour_bboxes.size == 0:
            new_contours = []
            # если движения нет, то проверяем по истории, есть ли люди в кадре
            for hist_bbox in last_hist_bboxes:
                if self.get_ssd_prediction()[2] > 0.5: # если conf больше порогового, то закидываем человека в итоговый ббокс
                    new_contours.append(hist_bbox)
        else:
            # Если движение есть и чтобы никого не потерять, создаем уникальные комбинации для перебора
            # вида [[(ббокс_истории1, контур1), (ббокс_истории2, контур1), ..], ..]
            unique_combinations = list(list(zip(last_hist_bboxes, element))
                                       for element in product(contour_bboxes, repeat=len(last_hist_bboxes)))
            # нужно найти и отсеять по пороговому iou, а тех, у кого ниже - проверить и добавить в стек,
            # и его уже закинуть в трекер (так, как это было реализовано с min max ранее)
            for combination in unique_combinations:
                for hist_bbox, contour_bbox in combination:
                    if get_iou(hist_bbox, contour_bbox) < 0.5:
                        np.vstack((np.array([]), [1, 1, 1, 1])) # для закидывания нового массива в стек

    def get_ssd_prediction(self, cropped_frame: np.array) -> np.array:
        """..."""
        blob = cv2.dnn.blobFromImage(
                cropped_frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        self.ssd_net.setInput(blob)
        detections = self.ssd_net.forward().squeeze()
        return detections[detections[:, 1] == 15]

    def update_tracks(self, tracks) -> None:
        """Обновляет треки, добавляя их в список с историей по id"""
        track_human_ids, track_human_bboxes = (tracks[:, -1].astype(int),  # id и bbox'ы
                                               tracks[:, :-1].astype(int))
        self.human_bbox_tracks = {
            human_id: human_bbox for human_id, human_bbox in self.human_bbox_tracks.items()
            if human_id in track_human_ids}  # фильтруем треки, которые потерялись
        for track_human_id, track_human_bbox in zip(track_human_ids, track_human_bboxes):
            # добавляем в историю трека или, если ее нет, создаем deque
            (self.human_bbox_tracks.setdefault(track_human_id, deque(maxlen=60)).
             append(np.array([track_human_bbox[:-2], track_human_bbox[2:]])))

    def processing_(self, original_frame: np.array):
        """"""
        # self.frame = frame
        contours, fg_mask = self.get_contours_fg_mask(original_frame)
        # распаковка из xywh в xyxy и фильтрация по площади
        filtered_contour_bboxes = np.fromiter(map(
                lambda xywh: np.array([xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]),
                [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > self.min_area]
        ), dtype=np.dtype((int, 4)))
        # self.check_contours(filtered_contour_bboxes)
        if filtered_contour_bboxes.size != 0:
            bboxes_to_track = np.concatenate(
                    [filtered_contour_bboxes, np.stack([[0.7]] * filtered_contour_bboxes.shape[0])], axis=1)
        else:
            bboxes_to_track = np.empty((0, 5))
        self.update_tracks(self.tracker.update(bboxes_to_track))
        return fg_mask, self.human_bbox_tracks


if __name__ == '__main__':
    cap = cv2.VideoCapture('new_office_test.mp4')
    _, frame = cap.read()
    back_sub_shape = (np.array(frame.shape[:-1][::-1]) / 2).astype(int)
    processing = FrameProcessing(frame.shape)
    first = []
    second = []
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break
        # frame = cv2.resize(frame, back_sub_shape)
        fgmask, bbox_tracks = processing.processing_(frame)
        # Create a copy of the frame to draw bounding boxes around the detected cars.
        frameCopy = frame.copy()

        for idx, bboxes in bbox_tracks.items():
            x1, y1, x2, y2 = np.concatenate(bboxes[-1])
            if idx == 2:
                first.append([x1 + x2 / 2, y1 + y2 / 2])
            if idx == 4:
                second.append([x1 + x2 / 2, y1 + y2 / 2])
            cv2.putText(frameCopy, str(idx), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frameCopy, bboxes[-1][0].astype(int), bboxes[-1][1].astype(int), (0, 255, 0), 2)

        # Extract the foreground from the frame using the segmented mask.
        foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)

        # out.write(frameCopy)
        cv2.imshow('main', cv2.resize(frameCopy, back_sub_shape))

        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
