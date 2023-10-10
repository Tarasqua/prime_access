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
        """..."""
        cap = cv2.VideoCapture(self.stream_source)
        _, frame = cap.read()
        assert frame is not None
        frame_shape = frame.shape
        frame_dtype = frame.dtype
        roi = ROIPolygon().get_roi(cv2.resize(frame, (np.array(frame.shape[:-1][::-1]) / 2).astype(int))) * 2
        self.detector = EntranceDetector(frame_shape, frame_dtype, roi, True)
        while cap.isOpened():
            _, self.frame = cap.read()
            self.frame = await self.detector.detect_(self.frame)
            if self.show_stream:
                cv2.imshow('det',
                           cv2.resize(self.frame, (np.array(self.frame.shape[:-1][::-1]) / 2).astype(int)))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main = Main('new_office_test.mp4')
    asyncio.run(main.main())
