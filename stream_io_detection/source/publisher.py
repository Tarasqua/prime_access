"""..."""

import multiprocessing
import pickle

from utils.templates import PreprocessedPerson
from propan import RabbitBroker
from propan.brokers.rabbit import RabbitExchange, ExchangeType, RabbitQueue
from io_config_loader import IOConfig


class Publisher:
    """Передатчик данных на брокер"""

    def __init__(self):
        config_ = IOConfig('config_rabbit.yml')
        self.routing_key = config_.get('DEBUG_ENTRANCE', 'ROUTING_KEY')
        self.queue = RabbitQueue(config_.get('DEBUG_ENTRANCE', 'QUEUE_NAME'),
                                 routing_key=f"*.{self.routing_key}")
        self.exchange = RabbitExchange(config_.get('DEBUG_ENTRANCE', 'TOPIC_EXCHANGE_NAME'),
                                       type=ExchangeType.TOPIC)

    async def publish_(self, data: PreprocessedPerson):
        """
        Передача единичных данных в очередь сообщений.
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
