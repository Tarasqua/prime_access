import datetime
import os
import itertools
from pathlib import Path
import asyncio
from collections import deque

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from plots import Plots
import line_zone_custom
from tracker import Tracker


class Main:

    def __init__(self, stream_source: int | str = 0, yolo_model: str = 'n'):
        self.stream_source = stream_source

        self.plots = Plots()
        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_seg_model = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-seg.onnx'))
        # self.yolo_detect_model = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}.onnx'))
        self.ssd_net = cv2.dnn.readNetFromCaffe(
            os.path.join(models_path, 'ssd_models', 'MobileNetSSD_deploy.prototxt'),
            os.path.join(models_path, 'ssd_models', 'MobileNetSSD_deploy.caffemodel')
        )
        self.orig_frame = None
        self.frame = None
        self.human_bbox_tracks = {}
        self.tracker = Tracker(max_age=90, min_hits=1, iou_threshold=0.3)
        # self.left_point, self.right_point = sv.Point(291, 546), sv.Point(1763, 472)
        self.left_point, self.right_point = sv.Point(579, 169), sv.Point(1559, 368)
        self.line_counter = line_zone_custom.LineZone(start=self.right_point, end=self.left_point)
        self.line_annotator = line_zone_custom.LineZoneAnnotator(
            thickness=2, text_thickness=1, text_scale=0.5)

        self.yolo_confidence = 0.5
        self.bbox_confidence = 0.5
        self.area_threshold = 1000
        self.iou_threshold = 0.3
        self.main_color = (0, 255, 0)

    @staticmethod
    def click_event(event, x, y, flags, params) -> None:
        """Кликер для определения координат для построения линии пересечения"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'[{datetime.datetime.now().time().strftime("%H:%M:%S")}]: ({x}, {y})')

    @staticmethod
    def get_video_writer(cap):
        records_folder_path = os.path.join(Path(__file__).resolve().parents[1], 'records')
        if not os.path.exists(records_folder_path):
            os.mkdir(records_folder_path)
        out = cv2.VideoWriter(
            os.path.join(records_folder_path, f'{len(os.listdir(records_folder_path)) + 1}.mp4'),
            -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        return out

    def get_contours(self, back_sub, kernel) -> tuple:
        """Возвращает контуры из маски движения и маску переднего плана"""
        fg_mask = back_sub.apply(self.frame)  # маска переднего плана
        closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # морф. закрытие
        contours, hierarchy = cv2.findContours(  # находим контуры
            image=closing, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        return list(contours), closing

    @staticmethod
    async def crop_black(img_gray: np.ndarray) -> np.ndarray:
        """Обрезает черную рамку"""
        y_nonzero, x_nonzero = np.nonzero(img_gray)
        return img_gray[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    async def get_segmented_frame(self, frame):
        segm_result = self.yolo_seg_model.predict(frame, classes=[0], verbose=False, conf=0.5)[0]
        if segm_result.masks is None:
            return None
        mask = cv2.resize(
            segm_result.masks.data.numpy()[0].astype('uint8'), cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).shape[::-1])
        bitwise_gray = cv2.bitwise_and(
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), mask=mask)
        return bitwise_gray

    async def detect_face(self, trigger: dict):
        triggered_id, state = trigger.values()
        x1, y1, x2, y2 = np.concatenate(self.human_bbox_tracks[triggered_id][0])
        segmented_frame = await self.get_segmented_frame(self.frame[y1:y2, x1:x2, :])
        if segmented_frame is not None:
            cropped_bg = await self.crop_black(segmented_frame)
            cropped_face = await self.crop_black(cropped_bg[:int(cropped_bg.shape[0] / 3.5)])
            # cv2.imwrite(f'test{triggered_id}.png', cropped_face)

    async def update_tracks(self, tracks):
        track_human_ids, track_human_bboxes = tracks[:, -1].astype(int), tracks[:, :-1].astype(int)
        self.human_bbox_tracks = {human_id: human_bbox for human_id, human_bbox in self.human_bbox_tracks.items()
                                  if human_id in track_human_ids}  # фильтруем треки, которые потерялись
        for track_human_id, track_human_bbox in zip(track_human_ids, track_human_bboxes):
            if track_human_id not in self.human_bbox_tracks:
                self.human_bbox_tracks[track_human_id] = deque(maxlen=60)
                self.human_bbox_tracks[track_human_id].append(np.array([track_human_bbox[:-2], track_human_bbox[2:]]))
            else:
                self.human_bbox_tracks[track_human_id].append(np.array([track_human_bbox[:-2], track_human_bbox[2:]]))

    async def main(self):
        # cv2.namedWindow('main')
        # cv2.setMouseCallback('main', self.click_event)

        back_sub = cv2.bgsegm.createBackgroundSubtractorGSOC(
            noiseRemovalThresholdFacBG=0.005, noiseRemovalThresholdFacFG=0.01, propagationRate=0.2,
            blinkingSupressionDecay=0.001, replaceRate=0.1
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        cap = cv2.VideoCapture(self.stream_source)
        while True:
            _, frame = cap.read()
            if frame is not None:
                break
        self.yolo_seg_model.predict(frame, classes=[0], verbose=False, conf=0.5)
        # self.yolo_detect_model.predict(frame, classes=[0], conf=0.5)
        back_sub_shape = (np.array(frame.shape[:-1][::-1]) / 5).astype(int)
        # out = self.get_video_writer(cap)
        frame_counter = 1
        while cap.isOpened():
            start = datetime.datetime.now()
            _, self.orig_frame = cap.read()
            if self.orig_frame is None:
                break
            self.frame = cv2.resize(self.orig_frame, back_sub_shape)
            contours, fg_mask = self.get_contours(back_sub, kernel)
            contours = [contour for contour in contours  # фильтруем контуры по площади
                        if cv2.contourArea(contour) >= self.area_threshold]
            if contours:
                bboxes_to_track = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    # if frame_counter % 2 == 0:
                    blob = cv2.dnn.blobFromImage(
                        self.frame[y:y + h, x:x + w], 0.007843,
                        (300, 300), (127.5, 127.5, 127.5), False)
                    self.ssd_net.setInput(blob)
                    detections = self.ssd_net.forward().squeeze()
                    detections = detections[detections[:, 1] == 15]  # только человека
                    if any(detections[:, 2] > 0.5):
                        for detection in detections:
                            conf, x1, y1, x2, y2 = detection[2:]
                            # из относительных в абсолютные координаты bbox'а с человеком
                            x1, y1, x2, y2 = np.concatenate(
                                np.array([[x1, y1], [x2, y2]]) * self.frame[y:y + h, x:x + w].shape[:-1][::-1]
                            )
                            bboxes_to_track.append([int(x1 + x), int(y1 + y), int(x2 + x), int(y2 + y), conf])
                        continue
                    # if frame_counter % 2 == 0:
                    #     detections = self.yolo_detect_model.predict(
                    #         self.frame[y:y + h, x:x + w], classes=[0], conf=0.5, verbose=False)[0]
                    #     # detections = self.yolo_detect_model.predict(
                    #     #     self.frame, classes=[0], conf=0.5)[0]
                    #     if len(detections) != 0:
                    #         for bbox in detections.boxes.data.numpy():
                    #             x1, y1, x2, y2 = bbox[:-2].astype(int)
                    #             # с пересчетом в абсолютные координаты относительно изображения
                    #             bboxes_to_track.append(np.array([x1 + x, y1 + y, x2 + x, y2 + y, 1]))
                    #             # bboxes_to_track.append(np.array([x1, y1, x2, y2, 1]))
                    #         continue
                    bboxes_to_track.append([x, y, x + w, y + h, 0.7])
                bboxes_to_track = np.array(bboxes_to_track)
                tracks = self.tracker.update(bboxes_to_track)
            else:
                tracks = self.tracker.update()
            self.human_bbox_tracks = {human_id: np.array([human_bbox[:-2], human_bbox[2:]]) for human_id, human_bbox
                                      in zip(tracks[:, -1].astype(int), tracks[:, :-1].astype(int))}
            # await self.update_tracks(tracks)
            # if self.human_bbox_tracks:
            #     trigger = self.line_counter.trigger(self.human_bbox_tracks)
            #     if trigger is not None:
            #         await self.detect_face(trigger)
            # self.line_annotator.annotate(frame=self.frame, line_counter=self.line_counter)

            # out.write(fg_mask)
            if self.human_bbox_tracks:
                self.frame = await self.plots.plot_bboxes(self.frame, self.human_bbox_tracks, self.main_color)
            fps = np.round(1 / (datetime.datetime.now() - start).microseconds * 10 ** 6)
            cv2.putText(self.frame, f'fps: {fps}', (30, 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 1, cv2.FILLED)
            # cv2.imshow('fg_mask', cv2.resize(fg_mask, back_sub_shape))
            frame_counter += 1
            cv2.imshow('main', cv2.resize(self.frame, back_sub_shape))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # run = Main(stream_source='it_office.mp4')
    # run = Main(stream_source="rtsp://admin:Qwer123@192.168.9.126/cam/realmonitor?channel=1&subtype=0")
    run = Main(stream_source='../records/1.mp4')
    # run = Main(stream_source=0)
    asyncio.run(run.main())
