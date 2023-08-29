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
from misc import BackSubData, HumanDetectionData, LineZoneData


class Main:

    def __init__(self, stream_source: int | str = 0, yolo_model: str = 'n'):
        self.stream_source = stream_source

        self.plots = Plots()
        back_sub_data, human_det_data, line_zone_data = (
            BackSubData(), HumanDetectionData(), LineZoneData())
        models_path = os.path.join(Path(__file__).resolve().parents[1], 'models')
        self.yolo_seg_model = YOLO(os.path.join(models_path, 'yolo_models', f'yolov8{yolo_model}-seg.onnx'))
        ssd_models_path = os.path.join(models_path, 'ssd_models')
        self.ssd_net = cv2.dnn.readNetFromCaffe(
            os.path.join(ssd_models_path,  # берем файлы по расширению, чтобы не учитывать их название
                         [file for file in os.listdir(ssd_models_path) if file.endswith('.prototxt')][0]),
            os.path.join(ssd_models_path,
                         [file for file in os.listdir(ssd_models_path) if file.endswith('.caffemodel')][0])
        )
        self.back_sub_model = cv2.bgsegm.createBackgroundSubtractorGSOC(
            noiseRemovalThresholdFacBG=back_sub_data.gsoc_data.get('NOISE_REMOVAL_THRESHOLD_FAC_BG'),
            noiseRemovalThresholdFacFG=back_sub_data.gsoc_data.get('NOISE_REMOVAL_THRESHOLD_FAC_FG'),
            propagationRate=back_sub_data.gsoc_data.get('PROPAGATION_RATE'),
            blinkingSupressionDecay=back_sub_data.gsoc_data.get('BLINKING_SUPRESSION_DECAY'),
            replaceRate=back_sub_data.gsoc_data.get('REPLACE_RATE')
        )
        self.morph_closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(back_sub_data.morph_closing_data['KERNEL_SIZE']))
        self.morph_closing_iterations = back_sub_data.morph_closing_data['ITERATIONS']

        self.human_bbox_tracks = {}
        self.bboxes_to_track = []
        self.yolo_confidence = human_det_data.segm_model_confidence
        self.area_threshold = human_det_data.min_square_threshold
        self.tracker = Tracker(
            history_threshold=human_det_data.tracker_data['HISTORY_THRESHOLD'],
            min_hits=human_det_data.tracker_data['MIN_HITS'],
            iou_threshold=human_det_data.tracker_data['IOU_THRESHOLD'])

        self.line_counter = line_zone_custom.LineZone(
            start=sv.Point(*line_zone_data.right_point), end=sv.Point(*line_zone_data.left_point))
        self.line_annotator = line_zone_custom.LineZoneAnnotator(
            thickness=line_zone_data.line_thickness, text_thickness=line_zone_data.text_thickness,
            text_scale=line_zone_data.text_scale)

        self.orig_frame = None
        self.frame = None

    @staticmethod
    def click_event(event, x, y, flags, params) -> None:
        """Кликер для определения координат для построения линии пересечения"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'[{datetime.datetime.now().time().strftime("%H:%M:%S")}]: ({x}, {y})')

    @staticmethod
    def get_video_writer(cap) -> cv2.VideoWriter:
        """Возвращает объект VideoWriter'а для записи демо"""
        records_folder_path = os.path.join(Path(__file__).resolve().parents[1], 'records')
        if not os.path.exists(records_folder_path):
            os.mkdir(records_folder_path)
        out = cv2.VideoWriter(
            os.path.join(records_folder_path, f'{len(os.listdir(records_folder_path)) + 1}.mp4'),
            -1, 20.0, (int(cap.get(3)), int(cap.get(4))))
        return out

    def get_contours_fg_mask(self) -> tuple:
        """Возвращает контуры из маски движения и маску переднего плана"""
        fg_mask = self.back_sub_model.apply(self.frame)  # маска переднего плана
        closing = cv2.morphologyEx(  # морфологическое закрытие
            fg_mask, cv2.MORPH_CLOSE, self.morph_closing_kernel, iterations=self.morph_closing_iterations)
        contours, hierarchy = cv2.findContours(  # находим контуры
            image=closing, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        return list(contours), closing

    @staticmethod
    async def crop_black(img_gray: np.ndarray) -> np.ndarray:
        """Обрезает черную рамку"""
        y_nonzero, x_nonzero = np.nonzero(img_gray)
        return img_gray[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]

    async def get_segmented_frame(self, frame) -> np.ndarray:
        """Возвращает сегментированное изображение человека в кадре"""
        segm_result = self.yolo_seg_model.predict(frame, classes=[0], verbose=False, conf=self.yolo_confidence)[0]
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

    # Временная, но может пригодиться в будущем
    async def update_tracks(self, tracks) -> None:
        """Обновляет треки, добавляя их в список с историей по id"""
        track_human_ids, track_human_bboxes = tracks[:, -1].astype(int), tracks[:, :-1].astype(int)
        self.human_bbox_tracks = {human_id: human_bbox for human_id, human_bbox in self.human_bbox_tracks.items()
                                  if human_id in track_human_ids}  # фильтруем треки, которые потерялись
        for track_human_id, track_human_bbox in zip(track_human_ids, track_human_bboxes):
            (self.human_bbox_tracks.setdefault(track_human_id, deque(maxlen=60)).
             append(np.array([track_human_bbox[:-2], track_human_bbox[2:]])))

    async def get_detections_from_contour(self, contour) -> np.array:
        """
        Возвращает bbox контура и детекции людей в контуре в формате:
        [[_, class_name, confidence, x1, y1, x2, y2], ...]
        """
        x, y, w, h = cv2.boundingRect(contour)
        blob = cv2.dnn.blobFromImage(
            self.frame[y:y + h, x:x + w], 0.007843,
            (300, 300), (127.5, 127.5, 127.5), False)
        self.ssd_net.setInput(blob)
        detections = self.ssd_net.forward().squeeze()
        return x, y, w, h, detections[detections[:, 1] == 15]  # только человека

    async def process_contour(self, contour):
        x, y, w, h, detections = await self.get_detections_from_contour(contour)
        if any(detections[:, 2] > 0.5):
            for detection in detections:
                conf, x1, y1, x2, y2 = detection[2:]
                # из относительных в абсолютные координаты bbox'а с человеком
                x1, y1, x2, y2 = np.concatenate(
                    np.array([[x1, y1], [x2, y2]]) * self.frame[y:y + h, x:x + w].shape[:-1][::-1]
                )
                self.bboxes_to_track.append([x1 + x, y1 + y, x2 + x, y2 + y, conf])
        else:
            self.bboxes_to_track.append([x, y, x + w, y + h, 0.7])

    async def main(self):
        cv2.namedWindow('main')
        cv2.setMouseCallback('main', self.click_event)

        cap = cv2.VideoCapture(self.stream_source)
        _, frame = cap.read()
        self.yolo_seg_model.predict(frame, classes=[0], verbose=False, conf=self.yolo_confidence)
        back_sub_shape = (np.array(frame.shape[:-1][::-1]) / 6).astype(int)
        # out = self.get_video_writer(cap)
        while cap.isOpened():
            start = datetime.datetime.now()
            _, self.orig_frame = cap.read()
            if self.orig_frame is None:
                break
            self.frame = cv2.resize(self.orig_frame, back_sub_shape)
            contours, fg_mask = self.get_contours_fg_mask()
            contours = [contour for contour in contours  # фильтруем контуры по площади
                        if cv2.contourArea(contour) >= self.area_threshold]
            self.bboxes_to_track = []
            if contours:
                bboxes_tasks = []
                for contour in contours:
                    bboxes_tasks.append(asyncio.create_task(self.process_contour(contour)))
                for bboxes_task in bboxes_tasks:
                    await bboxes_task
                tracks = self.tracker.update(np.array(self.bboxes_to_track))
            else:
                tracks = self.tracker.update()

            self.human_bbox_tracks = {  # в абсолютных координатах оригинального изображения
                human_id: ((np.array([human_bbox[:-2], human_bbox[2:]]) /
                           self.frame.shape[:-1][::-1]) * self.orig_frame.shape[:-1][::-1]).astype(int)
                for human_id, human_bbox in zip(tracks[:, -1].astype(int), tracks[:, :-1])}
            # await self.update_tracks(tracks)
            if self.human_bbox_tracks:
                trigger = self.line_counter.trigger(self.human_bbox_tracks)
                # if trigger is not None:
                #     await self.detect_face(trigger)
            self.line_annotator.annotate(frame=self.orig_frame, line_counter=self.line_counter)

            if self.human_bbox_tracks:
                self.orig_frame = await self.plots.plot_bboxes(self.orig_frame, self.human_bbox_tracks)
            fps = np.round(1 / (datetime.datetime.now() - start).microseconds * 10 ** 6)
            cv2.putText(self.orig_frame, f'fps: {fps}', (30, 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 1, cv2.FILLED)
            # cv2.imshow('fg_mask', cv2.resize(fg_mask, back_sub_shape))
            # out.write(self.orig_frame)
            cv2.imshow('main', self.orig_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # run = Main(stream_source='it_office.mp4')
    # run = Main(stream_source="rtsp://admin:Qwer123@192.168.9.126/cam/realmonitor?channel=1&subtype=0")
    run = Main(stream_source='office.mp4')
    # run = Main(stream_source=0)
    asyncio.run(run.main())
