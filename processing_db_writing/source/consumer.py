"""
Реализация брокера сообщений
"""
import asyncio
import pickle

from propan import PropanApp, RabbitBroker
from propan.annotations import Logger
from propan.brokers.rabbit import ExchangeType, RabbitExchange, RabbitQueue

from utils.templates import PreprocessedPerson
from transmitter import Transmitter
from processing_config_loader import ProcessingConfig


config_ = ProcessingConfig('config_rabbit.yml')
broker = RabbitBroker(config_.get('DEBUG_ENTRANCE', 'HOST'))
app = PropanApp(broker)
topic_exchange = RabbitExchange(config_.get('DEBUG_ENTRANCE', 'TOPIC_EXCHANGE_NAME'),
                                type=ExchangeType.TOPIC)
queue = RabbitQueue(config_.get('DEBUG_ENTRANCE', 'QUEUE_NAME'),
                    routing_key=f"*.{config_.get('DEBUG_ENTRANCE', 'ROUTING_KEY')}")
transmitter = Transmitter()


@broker.handle(queue, topic_exchange)
async def entrance_handler(body: bytes, logger: Logger):
    """
    Обработка данных из очереди.
    Parameters:
        body: сообщение из очереди в байтах
        logger: логгер событий
    """
    entrance_data: PreprocessedPerson = pickle.loads(body)
    save_directory: str = await transmitter.transmit_(entrance_data)
    logger.info(f"Data saved in: {save_directory}")  # временно отписываем о сохранении данных


if __name__ == "__main__":
    asyncio.run(app.run())
