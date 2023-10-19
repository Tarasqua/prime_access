"""..."""
import asyncio
import datetime

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

    async def main(self):
        """Временная имплементация мейна для тестов"""
        cap = cv2.VideoCapture(self.stream_source)
        _, frame = cap.read()
        assert frame is not None, "Couldn't open stream source"
        roi = ROIPolygon().get_roi(cv2.resize(frame, (np.array(frame.shape[:-1][::-1]) / 2).astype(int))) * 2
        self.detector = EntranceDetector(frame.shape, frame.dtype, roi)
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = Main('../../resources/videos/new_office_test.mp4')
    # main = Main('../resources/videos/crowd.mp4')
    asyncio.run(main.main())
