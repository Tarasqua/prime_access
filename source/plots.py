import asyncio
import cv2
import numpy as np


class Plots:
    """..."""
    def __init__(self, kpts_conf: float):
        self.kpts_conf = kpts_conf
        self.frame = None

        key_points = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                      'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                      'right_knee', 'left_ankle', 'right_ankle']
        self.limbs = [
            [key_points.index('right_eye'), key_points.index('nose')],
            [key_points.index('right_eye'), key_points.index('right_ear')],
            [key_points.index('left_eye'), key_points.index('nose')],
            [key_points.index('left_eye'), key_points.index('left_ear')],
            [key_points.index('right_shoulder'), key_points.index('right_elbow')],
            [key_points.index('right_elbow'), key_points.index('right_wrist')],
            [key_points.index('left_shoulder'), key_points.index('left_elbow')],
            [key_points.index('left_elbow'), key_points.index('left_wrist')],
            [key_points.index('right_hip'), key_points.index('right_knee')],
            [key_points.index('right_knee'), key_points.index('right_ankle')],
            [key_points.index('left_hip'), key_points.index('left_knee')],
            [key_points.index('left_knee'), key_points.index('left_ankle')],
            [key_points.index('right_shoulder'), key_points.index('left_shoulder')],
            [key_points.index('right_hip'), key_points.index('left_hip')],
            [key_points.index('right_shoulder'), key_points.index('right_hip')],
            [key_points.index('left_shoulder'), key_points.index('left_hip')]
        ]
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])
        self.pose_limb_color = palette[[16, 16, 16, 16, 9, 9, 9, 9, 0, 0, 0, 0, 7, 7, 7, 7]]
        self.pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

    async def __plot_person_kpts(self, kpts) -> None:
        """Строит ключевые точки и суставы скелета человека"""
        for p_id, point in enumerate(kpts):
            x_coord, y_coord, conf = point
            if conf < self.kpts_conf:
                continue
            r, g, b = self.pose_kpt_color[p_id]
            cv2.circle(self.frame, (int(x_coord), int(y_coord)), 5, (int(r), int(g), int(b)), -1)
        for sk_id, sk in enumerate(self.limbs):
            r, g, b = self.pose_limb_color[sk_id]
            if kpts[sk[0]][2] < self.kpts_conf or \
                    kpts[sk[1]][2] < self.kpts_conf:
                continue
            pos1 = int(kpts[sk[0]][0]), int(kpts[sk[0]][1])
            pos2 = int(kpts[sk[1]][0]), int(kpts[sk[1]][1])
            cv2.line(self.frame, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

    async def __plot_person_bbox(self, bbox, person_id):
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(self.frame, str(person_id), (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

    async def plot_kpts(self, frame: np.array, detections) -> np.array:
        """Отрисовка bbox'ов людей и их id"""
        self.frame = frame
        skeleton_tasks = []
        bboxes_tasks = []
        for detection in detections:
            skeleton_tasks.append(asyncio.create_task(
                self.__plot_person_kpts(detection.keypoints.data.numpy()[0])
            ))
            bboxes_tasks.append(asyncio.create_task(
                self.__plot_person_bbox(detection.boxes.xyxy.data.numpy()[0], detection.boxes.id.data.numpy()[0])
            ))
            if detection.keypoints.data.numpy()[0][5][-1] < 0.8 or detection.keypoints.data.numpy()[0][6][-1] < 0.8:
                continue
            x, y = detection.keypoints.data.numpy()[0][0][:-1].astype(int)
            if detection.keypoints.data.numpy()[0][6][0] < detection.keypoints.data.numpy()[0][5][0]:
                cv2.putText(self.frame, 'leaving', (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))
            else:
                cv2.putText(self.frame, 'entering', (x, y - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))
        for skeleton_task in skeleton_tasks:
            await skeleton_task
        for bboxes_task in bboxes_tasks:
            await bboxes_task
        return self.frame
