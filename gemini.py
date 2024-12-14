import asyncio
import json
import logging
from typing import Optional, List, Dict
from httpsrq import ModelConfig

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini.log"),
        logging.StreamHandler()
    ]
)



class BaseGenaiRequest:
    """Базовый класс для работы с генеративной моделью."""
    def __init__(
        self,
        api_key: str,
        prompt: str = "",
        file_path: Optional[List[str]] = None,
        history: Optional[List[Dict]] = None,
        model: str = "gemini-1.5-flash",
        system_instruction: str = "You are an assistant",
        max_output_tokens: int = 8192,
        temperature: float = 1.0,
    ):
        self.api_key = api_key
        self.prompt = prompt
        self.file_path = file_path
        self.history = history or []
        self.model = model
        self.system_instruction = system_instruction
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    async def __call__(self) -> Dict:
        """Вызов генерации текста."""
        logging.info("Создание ModelConfig с параметрами: %s", {
            "model": self.model,
            "instruction": self.system_instruction,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        })
        model = ModelConfig(
            model_name=self.model,
            system_instruction=self.system_instruction,
            generation_config={
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            },
            api_key=self.api_key,
        )

        chat = model.start_chat(history=self.history)
        try:
            response = await chat.send_message(content=self.prompt, images=self.file_path)
            logging.info("Успешный запрос. Ответ: %s", response)
            return response
        except Exception as e:
            logging.error("Ошибка при вызове модели: %s", e)
            raise

    def __await__(self):
        """Сделать объект вызываемым."""
        return self().__await__()


async def main():
    request = BaseGenaiRequest(
        api_key="AIzaSyAqlXbKAUBSwmnBV0KY0OehCGJVeaF-4fk",
        prompt="say Meow",
        system_instruction="You are Neko the cat, respond like one",
        # file_path=["111.jpg"]
    )
    response = await request
    print(json.dumps(response, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
