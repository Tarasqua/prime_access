"""..."""
import os
import asyncio
from pathlib import Path

import cv2
import torch

from utils.templates import PreprocessedPerson
from processing_config_loader import ProcessingConfig
from face_classification import FaceClassifier


class Transmitter:
    """Общение между базой данных и классификацией человека"""

    def __init__(self):
        config_ = ProcessingConfig('config_processing.yml')
        self.face_detector = FaceClassifier(config_.get('FACE_DETECTION'))

    async def transmit_(self, data: PreprocessedPerson):
        """
        Тестовая реализация без полноценной классификации лиц и общения с базой данных.
        """
        preprocessed_images = await self.face_detector.classify_(data)
        data.detection_frames = preprocessed_images
        return await self.save_data(data)

    @staticmethod
    async def save_data(data: PreprocessedPerson):
        """
        Временно для визуализации данных. Сохраняет изображения.
        Parameters:
            data: данные по человеку в формате PreprocessedPerson
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


if __name__ == '__main__':
    # ДЛЯ ОТЛАДКИ
    t = Transmitter()
    asyncio.run(t.transmit_(torch.load('preprocessed_.pt')))
