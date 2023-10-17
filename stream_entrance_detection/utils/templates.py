"""..."""
import datetime
from collections import deque
from pydantic import BaseModel


class TrackingPerson(BaseModel):
    """
    Структура данных для затреченного человека
    Parameters:
        frames_counter: сколько кадров наблюдается движение трека
        detection_frames: кропнутые кадры трека
        centroid_coordinates: координаты центроида для уточнения - шел человек или стоял на месте и двигался
        is_entering_statistics: статистика по детекции входа/выхода по координатам плеч
    """
    frames_counter: int = 0
    detection_frames: deque = deque(maxlen=25)
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
                case 'centroid_coordinates':
                    self.centroid_coordinates.append(value)
                case 'is_entering_statistics':
                    self.is_entering_statistics.append(value)


class PreprocessedPerson(BaseModel):
    """
    Структура данных для предобработанного человека
    Parameters:
        detection_frames: кропнутые кадры трека
        has_entered: вошел (True) или вышел (False) человек
        detection_time: время обнаружения
    """
    detection_frames: deque = deque(maxlen=25)
    has_entered: bool = True
    detection_time: datetime.datetime = datetime.datetime.now()
