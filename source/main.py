import datetime
import os
import itertools
from pathlib import Path
import asyncio

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
        self.frame = None
        self.human_bbox = {}
        self.human_id_counter = itertools.count()
        # self.left_point, self.right_point = sv.Point(291, 546), sv.Point(1763, 472)
        self.left_point, self.right_point = sv.Point(339, 343), sv.Point(1526, 344)
        self.line_counter = line_zone_custom.LineZone(start=self.right_point, end=self.left_point)
        self.line_annotator = line_zone_custom.LineZoneAnnotator(
            thickness=2, text_thickness=1, text_scale=0.5)

        self.yolo_confidence = 0.5
        self.bbox_confidence = 0.5
        self.area_threshold = 10000
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

    def get_contours(self, back_sub, back_sub_shape, kernel) -> tuple:
        """Возвращает контуры из маски движения и маску переднего плана"""
        fg_mask = back_sub.apply(cv2.resize(self.frame, back_sub_shape))  # маска переднего плана
        closing = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # морф. закрытие
        closing = cv2.resize(closing, self.frame.shape[:-1][::-1])  # ресайзим в исходный размер
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
        x1, y1, x2, y2 = np.concatenate(self.human_bbox[triggered_id][0])
        segmented_frame = await self.get_segmented_frame(self.frame[y1:y2, x1:x2, :])
        if segmented_frame is not None:
            cropped_bg = await self.crop_black(segmented_frame)
            cropped_face = await self.crop_black(cropped_bg[:int(cropped_bg.shape[0] / 3.5)])
            # cv2.imwrite(f'test{triggered_id}.png', cropped_face)

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
        back_sub_shape = (np.array(frame.shape[:-1][::-1]) / 4).astype(int)
        # out = self.get_video_writer(cap)
        tracker = Tracker(max_age=1, min_hits=3, iou_threshold=0.3)
        while cap.isOpened():
            start = datetime.datetime.now()
            _, self.frame = cap.read()
            if self.frame is None:
                break
            contours, fg_mask = self.get_contours(back_sub, back_sub_shape, kernel)
            if contours:
                contours = [contour for contour in contours if cv2.contourArea(contour) >= self.area_threshold]
                bboxes = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    bboxes.append([x, y, x + w, y + h, 1])
                bboxes = np.array(bboxes)
                tracks = tracker.update(bboxes)
            else:
                tracks = tracker.update()

            self.human_bbox = {human_id: np.array([human_bbox[:-2], human_bbox[2:]]) for human_id, human_bbox
                               in zip(tracks[:, -1].astype(int), tracks[:, :-1].astype(int))}
            if self.human_bbox:
                trigger = self.line_counter.trigger(self.human_bbox)
                if trigger is not None:
                    await self.detect_face(trigger)
            self.line_annotator.annotate(frame=self.frame, line_counter=self.line_counter)

            # out.write(fg_mask)
            self.frame = await self.plots.plot_bboxes(self.frame, self.human_bbox, self.main_color)
            fps = np.round(1 / (datetime.datetime.now() - start).microseconds * 10 ** 6)
            cv2.putText(self.frame, f'fps: {fps}', (30, 30), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 255, 0), 1, cv2.FILLED)
            # cv2.imshow('main', self.frame)
            cv2.imshow('main', self.frame)
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
