import asyncio
import pickle
import logging

from propan import PropanApp, RabbitBroker
from propan.annotations import Logger
from propan.brokers.rabbit import ExchangeType, RabbitExchange, RabbitQueue
from propan.annotations import ContextRepo
from pydantic_settings import BaseSettings

from utils.templates import PreprocessedPerson
from transmitter import Transmitter


broker = RabbitBroker("amqp://tarasqua:14021993@localhost:5672/")
app = PropanApp(broker)
topic_exchange = RabbitExchange("entrance", type=ExchangeType.TOPIC)
queue = RabbitQueue("entrance", routing_key="*.debug")
transmitter = Transmitter()


@broker.handle(queue, topic_exchange)
async def entrance_handler(body: bytes, logger: Logger):
    entrance_data: PreprocessedPerson = pickle.loads(body)
    save_directory: str = await transmitter.transmit_(entrance_data)
    logger.info(f"Data saved in: {save_directory}")


if __name__ == "__main__":
    asyncio.run(app.run())
