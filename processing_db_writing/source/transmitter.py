"""..."""
import os
from pathlib import Path

import cv2

from utils.templates import PreprocessedPerson


class Transmitter:
    """Общение между базой данных и классификацией человека"""

    def __init__(self):
        pass

    @staticmethod
    async def transmit_(data: PreprocessedPerson) -> str:
        """
        Тестовая реализация
        """
        detections_folder_path = os.path.join(Path(__file__).resolve().parents[2], 'resources', 'detections')
        if data.has_entered:
            entering_dir_path = os.path.join(detections_folder_path, 'entering')
            save_directory = os.path.join(entering_dir_path, str(data.person_id))
            os.mkdir(save_directory)
        else:
            leaving_dir_path = os.path.join(detections_folder_path, 'leaving')
            save_directory = os.path.join(leaving_dir_path, str(data.person_id))
            os.mkdir(save_directory)
        [cv2.imwrite(os.path.join(save_directory, f'{idx}.png'), picture)
         for (idx, picture) in enumerate(data.detection_frames)]
        return save_directory
