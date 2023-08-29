import os
import yaml
from pathlib import Path


class ConfigNotFoundException(Exception):
    """Ошибка, возникающая при отсутствии config файла."""

    def __str__(self):
        return "\nCouldn't find config file"


class BackSubData:
    """Данные из config по BACK_SUB_DATA"""

    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.gsoc_data: dict = self.config['BACK_SUB_DATA']['GSOC_DATA']
        self.morph_closing_data: dict = self.config['BACK_SUB_DATA']['MORPH_CLOSING_DATA']


class HumanDetectionData:
    """Данные из config по HUMAN_DETECTION_DATA"""

    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.min_square_threshold: int = self.config['HUMAN_DETECTION_DATA']['MIN_SQUARE_THRESHOLD']
        self.segm_model_confidence: float = self.config['HUMAN_DETECTION_DATA']['SEGM_MODEL_CONFIDENCE']
        self.tracker_data: dict = self.config['HUMAN_DETECTION_DATA']['TRACKER_DATA']


class LineZoneData:
    """Данные из config по LINE_ZONE_DATA"""

    def __init__(self):
        config_path = os.path.join(Path(__file__).resolve().parents[1], 'config', 'config.yaml')
        if not os.path.exists(config_path):
            raise ConfigNotFoundException
        with open(config_path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.left_point: tuple = tuple(self.config['LINE_ZONE_DATA']['LEFT_POINT'])
        self.right_point: tuple = tuple(self.config['LINE_ZONE_DATA']['RIGHT_POINT'])
        self.line_thickness: int = self.config['LINE_ZONE_DATA']['LINE_THICKNESS']
        self.text_thickness: int = self.config['LINE_ZONE_DATA']['TEXT_THICKNESS']
        self.text_scale: float = self.config['LINE_ZONE_DATA']['TEXT_SCALE']

