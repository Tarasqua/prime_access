from typing import Dict

import cv2
import numpy as np

from supervision.draw.color import Color
from supervision.geometry.core import Point, Rect, Vector


class LineZone:
    """
    Count the number of objects that cross a line.
    """

    def __init__(self, start: Point, end: Point):
        """
        Initialize a LineCounter object.

        Attributes:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.

        """
        self.vector = Vector(start=start, end=end)
        self.tracker_state: Dict[str, (bool, bool)] = {}  # 1 флаг - внутри или снаружи, второй - сменил состояние
        self.in_count: int = 0
        self.out_count: int = 0

    def trigger(self, human_bbox: dict):
        """
        Обновляет in_count и out_count для детекций, которые пересекли линию.
        Атрибуты:
            human_bbox: словарь из tracker_id: [xyxy, True/False], где True/False - тречится или уже нет
        """
        for tracker_id, bbox in human_bbox.items():
            x1, y1, x2, y2 = np.concatenate(bbox)
            # по центроиду смотрим, что человек зашел или вышел, а также, чтобы устранить ложные сработки из-за
            # скачков рамки, смотрим, чтобы человек не мог пройти тут же 2 раза, а только после того, как полностью
            # пересечет линию своим bbox'ом
            if self.tracker_state.get(tracker_id) is not None and self.tracker_state[tracker_id][-1]:
                triggers = [self.vector.is_in(point=anchor) for anchor
                            in [Point(x1, y1), Point(x1, y2), Point(x2, y1), Point(x2, y2)]]
                if len(set(triggers)) != 2:
                    self.tracker_state[tracker_id] = self.tracker_state[tracker_id][0], False
            if self.tracker_state.get(tracker_id) is not None and self.tracker_state[tracker_id][-1]:
                continue
            upper_middle_centroid = np.array([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 4]).astype('int32')
            # upper_centroid = np.array([x1 + (x2 - x1) / 2, y1]).astype('int32')
            trigger = self.vector.is_in(point=Point(*upper_middle_centroid))  # центроид внутри или снаружи
            # если данного id еще нет в базе, он добавляется и отслеживается его первое вхождение -
            # внутри он был изначально или снаружи
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = trigger, False
                continue
            # если объект остается внутри/снаружи, то скипаем его
            if self.tracker_state.get(tracker_id)[0] == trigger:
                continue
            # меняем флаг на противоположный, если человек сменил состояние
            self.tracker_state[tracker_id] = trigger, True
            # если был False и стал True, то человек вошел, и наоборот
            if trigger:
                self.in_count += 1
            else:
                self.out_count += 1
            return {"id": tracker_id, "state": trigger}


class LineZoneAnnotator:
    def __init__(
            self,
            thickness: float = 2,
            color: Color = Color.white(),
            text_thickness: int = 2,
            text_color: Color = Color.black(),
            text_scale: float = 0.5,
            text_offset: float = 1.5,
            text_padding: int = 10,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        Attributes:
            thickness (float): The thickness of the line that will be drawn.
            color (Color): The color of the line that will be drawn.
            text_thickness (int): The thickness of the text that will be drawn.
            text_color (Color): The color of the text that will be drawn.
            text_scale (float): The scale of the text that will be drawn.
            text_offset (float): The offset of the text that will be drawn.
            text_padding (int): The padding of the text that will be drawn.

        """
        self.thickness: float = thickness
        self.color: Color = color
        self.text_thickness: int = text_thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding

    def annotate(self, frame: np.ndarray, line_counter: LineZone) -> np.ndarray:
        """
        Draws the line on the frame using the line_counter provided.

        Attributes:
            frame (np.ndarray): The image on which the line will be drawn.
            line_counter (LineCounter): The line counter that will be used to draw the line.

        Returns:
            np.ndarray: The image with the line drawn on it.

        """
        cv2.line(frame, line_counter.vector.start.as_xy_int_tuple(), line_counter.vector.end.as_xy_int_tuple(),
                 self.color.as_bgr(), self.thickness, lineType=cv2.LINE_AA, shift=0)
        cv2.circle(frame, line_counter.vector.start.as_xy_int_tuple(), radius=5,
                   color=self.text_color.as_bgr(), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(frame, line_counter.vector.end.as_xy_int_tuple(), radius=5,
                   color=self.text_color.as_bgr(), thickness=-1, lineType=cv2.LINE_AA)

        in_text = f"in: {line_counter.in_count}"
        out_text = f"out: {line_counter.out_count}"

        (in_text_width, in_text_height), _ = cv2.getTextSize(
            in_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )
        (out_text_width, out_text_height), _ = cv2.getTextSize(
            out_text, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thickness
        )

        in_text_x = int((line_counter.vector.end.x + line_counter.vector.start.x - in_text_width) / 2)
        in_text_y = int((line_counter.vector.end.y + line_counter.vector.start.y + in_text_height) / 2
                        - self.text_offset * in_text_height)

        out_text_x = int((line_counter.vector.end.x + line_counter.vector.start.x - out_text_width) / 2)
        out_text_y = int((line_counter.vector.end.y + line_counter.vector.start.y + out_text_height) / 2
                         + self.text_offset * out_text_height)

        in_text_background_rect = Rect(x=in_text_x, y=in_text_y - in_text_height,
                                       width=in_text_width,height=in_text_height).pad(padding=self.text_padding)
        out_text_background_rect = Rect(x=out_text_x, y=out_text_y - out_text_height,
                                        width=out_text_width, height=out_text_height).pad(padding=self.text_padding)

        cv2.rectangle(frame, in_text_background_rect.top_left.as_xy_int_tuple(),
                      in_text_background_rect.bottom_right.as_xy_int_tuple(), self.color.as_bgr(), -1)
        cv2.rectangle(frame, out_text_background_rect.top_left.as_xy_int_tuple(),
                      out_text_background_rect.bottom_right.as_xy_int_tuple(), self.color.as_bgr(), -1)

        cv2.putText(frame, in_text, (in_text_x, in_text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale, self.text_color.as_bgr(), self.text_thickness, cv2.LINE_AA)
        cv2.putText(frame, out_text, (out_text_x, out_text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.text_scale, self.text_color.as_bgr(), self.text_thickness, cv2.LINE_AA)
