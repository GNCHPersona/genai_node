from typing import Optional, Dict, List
from misc import JsonDataBuilder
import aiohttp
import logging

# Настройка логгирования
logger = logging.getLogger(__name__)

class ModelConfig:
    """Настройка и конфигурация модели."""
    def __init__(
        self,
        model_name: str,
        system_instruction: Optional[str],
        generation_config: Optional[Dict],
        api_key: str,
        proxy: Optional[str] = None,
        safety_settings: Optional[Dict] = None
    ):
        if not api_key:
            raise ValueError("API key is required")
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.generation_config = generation_config
        self.api_key = api_key
        self.proxy = proxy
        self.safety_settings = safety_settings

    def start_chat(self, history: Optional[List[Dict]] = None) -> 'Message':
        return Message(self, history)


class Message:
    """Класс для отправки сообщений в API."""
    def __init__(self, model: ModelConfig, history: Optional[List[Dict]] = None) -> None:
        self.model = model
        self.history = history or []

    async def send_message(self, content: Optional[str] = None, images: Optional[List[str]] = None) -> Dict:
        """Отправка сообщения с поддержкой изображений."""
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.model.model_name}:generateContent'
        headers = {'Content-Type': 'application/json'}
        params = {'key': self.model.api_key}

        json_data = JsonDataBuilder(
            system_instruction=self.model.system_instruction,
            history=self.history,
            request_text=content,
            files=images,
        ).build()

        logger.info("Отправка POST-запроса на URL %s с данными: %s", url, json_data)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=url,
                    params=params,
                    headers=headers,
                    json=json_data,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("Успешный ответ от сервера: %s", data)
                        return data
                    else:
                        error_message = await response.text()
                        logger.error("Ошибка от API: %s, код ответа: %s", error_message, response.status)
                        return {"error": response.status, "message": error_message}
            except aiohttp.ClientError as e:
                logger.error("Ошибка клиента при запросе: %s", e)
                raise
