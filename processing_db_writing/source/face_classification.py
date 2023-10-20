import os
from pathlib import Path
import asyncio

import cv2
import numpy as np
from ultralytics import YOLO

from utils.templates import PreprocessedPerson


class FaceClassifier:
    """Классификация лица"""

    def __init__(self, config_data: dict):
        self.yolo_seg = self.__set_yolo_model(config_data['PREPROCESSING']['YOLO']['YOLO_MODEL'])
        self.yolo_confidence = config_data['PREPROCESSING']['YOLO']['YOLO_CONFIDENCE']

    @staticmethod
    def __set_yolo_model(yolo_model) -> YOLO:
        """
        Выполняет проверку путей и наличие модели:
            Если директория отсутствует, создает ее, а также скачивает в нее необходимую модель
        Parameters:
            yolo_model: n (nano), m (medium)...
        Returns:
            Объект YOLO-pose
        """
        yolo_models_path = os.path.join(Path(__file__).resolve().parents[2], 'resources', 'models', 'yolo_models')
        if not os.path.exists(yolo_models_path):
            Path(yolo_models_path).mkdir(parents=True, exist_ok=True)
        model_path = os.path.join(yolo_models_path, f'yolov8{yolo_model}-seg')
        if not os.path.exists(f'{model_path}.onnx'):
            YOLO(model_path).export(format='onnx')
        return YOLO(f'{model_path}.onnx')

    async def __preprocess_image(
            self, image: np.array, left_shoulder: np.array, right_shoulder: np.array) -> np.array:
        """
        Предобработка лиц для классификации:
            - сегментируем -> заливаем фон черным;
            - обрезаем изображение -> берем по плечам по горизонтали и снизу (сверху не трогаем);
            - переводим изображение в грейскейл.
        Parameters:
            image: изображение с человеком;
            left_shoulder: координаты левого плеча;
            right_shoulder: координаты правого плеча.
            По большому счету, порядок плеч не играет роли, так как все равно идет проверка на max и min значения.
        Returns:
            Предобработанное изображение.
        """
        prediction = self.yolo_seg.predict(image, classes=[0], verbose=False, conf=self.yolo_confidence)[0]
        mask = prediction.masks.data.numpy()[0]
        y_nz, x_nz = np.nonzero(mask)
        mask_wo_black = mask[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]  # обрезаем черные края маски
        segmented = cv2.bitwise_and(  # применяем маску к исходному изображению
            image, image, mask=cv2.resize(mask_wo_black, image.shape[:-1][::-1]).astype('uint8'))
        return cv2.cvtColor(
            segmented[:max(left_shoulder[1], right_shoulder[1]),
                      min(left_shoulder[0], right_shoulder[0]):max(left_shoulder[0], right_shoulder[0])],
            cv2.COLOR_BGR2GRAY)

    async def classify_(self, person_data: PreprocessedPerson):
        """
        Классификация лиц
        TODO: сделана только предобработка лица; доделать
        """
        preprocess_tasks = [
            asyncio.create_task(self.__preprocess_image(image, left_shoulder.astype(int), right_shoulder.astype(int)))
            for image, left_shoulder, right_shoulder
            in zip(person_data.detection_frames, person_data.left_shoulder, person_data.right_shoulder)]
        preprocessed_images = await asyncio.gather(*preprocess_tasks)
        return preprocessed_images
