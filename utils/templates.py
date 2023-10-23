"""..."""
import datetime
from collections import deque
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class TrackingPerson(BaseModel):
    """
    Структура данных для затреченного человека
    Parameters:
        frames_counter: сколько кадров наблюдается движение трека;
        detection_frames: кропнутые кадры трека;
        left_shoulder: координаты левого плеча
            (нужны для дальнейшей обработки изображения для классификации);
        right_shoulder: координаты правого плеча;
        centroid_coordinates: координаты центроида для уточнения -
            шел человек или стоял на месте и двигался;
        is_entering_statistics: статистика по детекции входа/выхода
            по координатам плеч.
    """
    frames_counter: int = 0
    detection_frames: deque = deque(maxlen=25)
    left_shoulder: deque = deque(maxlen=25)
    right_shoulder: deque = deque(maxlen=25)
    centroid_coordinates: deque = deque(maxlen=60)
    is_entering_statistics: deque = deque(maxlen=60)

    def update(self, **new_data):
        """Обновление данных"""
        for field, value in new_data.items():
            match field:
                case 'frames_counter':
                    self.frames_counter += value
                case 'detection_frames':
                    self.detection_frames.append(value)
                case 'left_shoulder':
                    self.left_shoulder.append(value)
                case 'right_shoulder':
                    self.right_shoulder.append(value)
                case 'centroid_coordinates':
                    self.centroid_coordinates.append(value)
                case 'is_entering_statistics':
                    self.is_entering_statistics.append(value)


class PreprocessedPerson(BaseModel):
    """
    Структура данных для предобработанного человека
    Parameters:
        person_id: уникальный id, который будет присваиваться при создании;
        detection_frames: кропнутые кадры трека;
        left_shoulder: координаты левого плеча
            (нужны для дальнейшей обработки изображения для классификации);
        right_shoulder: координаты правого плеча;
        has_entered: вошел (True) или вышел (False) человек;
        detection_time: время обнаружения.
    """
    person_id: UUID = Field(default_factory=uuid4)
    detection_frames: deque = deque(maxlen=25)
    left_shoulder: deque = deque(maxlen=25)
    right_shoulder: deque = deque(maxlen=25)
    has_entered: bool = True
    detection_time: datetime.datetime = datetime.datetime.now()


class ClassifiedPerson(BaseModel):
    """
    Структура данных для человека, прошедшего классфикацию
    Parameters:
        person_id: уникальный id;
        classification: ФИО работника (возможно, id);
        has_entered: вошел человек или нет;
        detection_time: время обнаружения.
    """
    person_id: UUID = Field(default_factory=uuid4)
    classification: str = ""
    has_entered: bool = True
    detection_time: datetime.datetime = datetime.datetime.now()
