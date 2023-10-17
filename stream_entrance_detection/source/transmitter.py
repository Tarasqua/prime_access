"""..."""

import os
import multiprocessing
from pathlib import Path

import cv2

from stream_entrance_detection.utils.templates import PreprocessedPerson


class Transmitter:
    """Передатчик данных на брокер"""

    def __init__(self):
        pass

    @staticmethod
    def transmit_(data: PreprocessedPerson):
        """
        Передача данных на брокер
        Parameters:
            data: предобработанные данные в формате PreprocessedPerson
        TODO: временное решение в виде простого сохранения заменить на отправку данных
        """
        detections_folder_path = os.path.join(Path(__file__).resolve().parents[1], 'resources', 'detections')
        if data.has_entered:
            entering_dir_path = os.path.join(detections_folder_path, 'entering')
            save_directory = os.path.join(entering_dir_path, str(len(os.listdir(entering_dir_path)) + 1))
            os.mkdir(save_directory)
        else:
            leaving_dir_path = os.path.join(detections_folder_path, 'leaving')
            save_directory = os.path.join(leaving_dir_path, str(len(os.listdir(leaving_dir_path)) + 1))
            os.mkdir(save_directory)
        [cv2.imwrite(os.path.join(save_directory, f'{idx}.png'), picture)
         for (idx, picture) in enumerate(data.detection_frames)]
        return save_directory

    @staticmethod
    def callback_(save_directory: str):
        """
        Callback передачи данных
        Parameters:
            save_directory: временное решение с отписью директории сохранения кадров
        TODO: сделать отпись в лог файл
        """
        name = multiprocessing.current_process().name
        print(f'{name} finished: pictures saved in {save_directory}')
