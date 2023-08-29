import asyncio
import cv2
import numpy as np


class Plots:

    def __init__(self):
        self.frame = None

    async def plot_bbox(self, human_id: int, bbox: np.array, color: tuple) -> None:
        """Отрисовка bbox'а и центроида человека"""
        cv2.putText(self.frame, str(human_id), bbox[0].astype(int), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 255, 0), 1, cv2.FILLED)
        cv2.rectangle(self.frame, bbox[0].astype(int), bbox[1].astype(int), color, 2)
        # center = np.array([bbox[0][0] + (bbox[1][0] - bbox[0][0]) / 2, bbox[0][1]]).astype(int)
        # cv2.circle(self.frame, center, 4, (0, 0, 255), -1)

    async def plot_bboxes(self, frame: np.array, human_bbox: dict, color: tuple = (0, 255, 0)) -> np.array:
        """Отрисовка bbox'ов людей и их id"""
        self.frame = frame
        bboxes_tasks = []
        for human_id, bbox in human_bbox.items():
            bboxes_tasks.append(asyncio.create_task(self.plot_bbox(human_id, bbox, color)))
        for bbox_task in bboxes_tasks:
            await bbox_task
        return self.frame
