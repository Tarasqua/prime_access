"""..."""
import asyncio
import datetime
import os

import cv2
import numpy as np

from entrance_detector import EntranceDetector
from roi_polygon import ROIPolygon


class Main:
    """..."""

    def __init__(self, stream_source: str, show_stream: bool = True):
        self.stream_source = stream_source
        self.show_stream = show_stream

        self.detector = None
        self.frame = None

    def __save_detections(self):
        for human_id, (pictures, is_entering, time) in self.detector.collected_data.items():
            if not os.path.exists(str(human_id)):
                os.mkdir(str(human_id))
            if is_entering:
                [cv2.imwrite(os.path.join(str(human_id), f'entering_{idx}.png'), picture)
                 for (idx, picture) in enumerate(pictures)]
            else:
                [cv2.imwrite(os.path.join(str(human_id), f'leaving_{idx}.png'), picture)
                 for (idx, picture) in enumerate(pictures)]

    async def main(self):
        """Временная имплементация мейна для тестов"""
        cap = cv2.VideoCapture(self.stream_source)
        _, frame = cap.read()
        assert frame is not None
        frame_shape = frame.shape
        frame_dtype = frame.dtype
        roi = ROIPolygon().get_roi(cv2.resize(frame, (np.array(frame.shape[:-1][::-1]) / 2).astype(int))) * 2
        # roi = np.array([[870, 200], [878, 548], [732, 680], [1156, 574], [1062, 504], [1058, 178]])
        self.detector = EntranceDetector(frame_shape, frame_dtype, roi)
        while cap.isOpened():
            start = datetime.datetime.now()
            _, self.frame = cap.read()
            if self.frame is None:
                break
            await self.detector.detect_(self.frame)
            if self.show_stream:
                frame_copy = self.frame.copy()
                fps = np.round(1 / (datetime.datetime.now() - start).microseconds * 10 ** 6)
                cv2.putText(frame_copy, f'fps: {fps}', (30, 30), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 0), 1, cv2.FILLED)
                cv2.polylines(
                    frame_copy, [roi], True, (60, 20, 220), 2)
                cv2.imshow('det',
                           cv2.resize(frame_copy, (np.array(self.frame.shape[:-1][::-1]) / 2).astype(int)))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        await self.detector.collect_data()
        self.__save_detections()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main = Main('new_office_test.mp4')
    main = Main('crowd.mp4')
    asyncio.run(main.main())
