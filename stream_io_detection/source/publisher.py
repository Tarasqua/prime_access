"""..."""

import multiprocessing
import pickle

from utils.templates import PreprocessedPerson
from propan import RabbitBroker
from propan.brokers.rabbit import RabbitExchange, ExchangeType, RabbitQueue


class Publisher:
    """Передатчик данных на брокер"""

    def __init__(self):
        self.routing_key = 'debug'
        self.queue = RabbitQueue("entrance", routing_key=f"*.{self.routing_key}")
        self.exchange = RabbitExchange("entrance", type=ExchangeType.TOPIC)

    async def publish_(self, data: PreprocessedPerson):
        """
        Передача данных на брокер
        Parameters:
            data: предобработанные данные в формате PreprocessedPerson
        """
        async with RabbitBroker() as broker:
            await broker.publish(
                message=pickle.dumps(data),
                routing_key=f"preprocessed_data.{self.routing_key}",
                queue=self.queue,
                exchange=self.exchange,
            )

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
