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

    async def main(self):
        """..."""
        cap = cv2.VideoCapture(self.stream_source)
        _, frame = cap.read()
        assert frame is not None
        frame_shape = frame.shape
        frame_dtype = frame.dtype
        roi = ROIPolygon().get_roi(cv2.resize(frame, (np.array(frame.shape[:-1][::-1]) / 2).astype(int))) * 2
        # roi = np.array([[870, 200], [878, 548], [732, 680], [1156, 574], [1062, 504], [1058, 178]])
        self.detector = EntranceDetector(frame_shape, frame_dtype, roi, True)
        while cap.isOpened():
            _, self.frame = cap.read()
            if self.frame is None:
                break
            self.frame = await self.detector.detect_(self.frame)
            if self.show_stream:
                cv2.polylines(
                    self.frame, [roi], True, (60, 20, 220), 2)
                cv2.imshow('det',
                           cv2.resize(self.frame, (np.array(self.frame.shape[:-1][::-1]) / 2).astype(int)))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        for human_id, (pictures, is_entering) in self.detector.collected_data.items():
            if not os.path.exists(str(human_id)):
                os.mkdir(str(human_id))
            if is_entering:
                for idx, picture in enumerate(pictures):
                    cv2.imwrite(os.path.join(str(human_id), f'entering_{idx}.png'), picture)
            else:
                for idx, picture in enumerate(pictures):
                    cv2.imwrite(os.path.join(str(human_id), f'leaving_{idx}.png'), picture)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # main = Main('new_office_test.mp4')
    main = Main('crowd.mp4')
    asyncio.run(main.main())
