from typing import Optional, Dict, List, Union
import os
import base64
import logging

logger = logging.getLogger(__name__)


class SystemInstruction:
    def __init__(self, system_instruction: Optional[str] = None) -> None:
        self.system_instruction = system_instruction

    def __call__(self) -> Optional[Dict[str, Dict[str, str]]]:
        if self.system_instruction:
            return {"parts": {"text": self.system_instruction}}
        return None


class Contents:
    """Форматирование содержимого для API."""
    def __init__(self, content: Optional[str] = None, images: Optional[List[str]] = None) -> None:
        self.contents = []

        if images:
            for image_path in images:
                mime_type = self._get_mime_type(image_path)
                encoded_data = self._encode_image(image_path)
                self.contents.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": encoded_data,
                    }
                })

        if content:
            self.contents.append({"text": content})

    @staticmethod
    def _get_mime_type(file_path: str) -> str:
        """Определяет MIME-тип файла на основе его расширения."""
        extension = os.path.splitext(file_path)[-1].lower()
        if extension in [".jpeg", ".jpg"]:
            return "image/jpeg"
        elif extension == ".png":
            return "image/png"
        else:
            logger.error("Unsupported file type: %s", extension)
            raise ValueError(f"Unsupported file type: {extension}")

    @staticmethod
    def _encode_image(file_path: str) -> str:
        """Кодирует файл в base64."""
        if not os.path.isfile(file_path):
            logger.error("Файл не найден: %s", file_path)
            raise ValueError(f"File not found: {file_path}")
        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
            logger.info("Файл успешно закодирован: %s", file_path)
            return encoded

    def __call__(self) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
        return self.contents


class JsonDataBuilder:
    """Сборка JSON-данных для API Gemini."""
    def __init__(
        self,
        system_instruction: Optional[str] = None,
        request_text: Optional[str] = None,
        history: Optional[List[Dict[str, Union[str, List[Dict[str, Union[str, Dict]]]]]]] = None,
        files: Optional[List[str]] = None,
    ) -> None:
        self.system_instruction = SystemInstruction(system_instruction)
        self.request_text = request_text
        self.history = history or []
        self.contents = Contents(content=request_text, images=files)

    def build(self) -> Dict:
        """Создает финальный JSON-объект для отправки."""
        json_data = {}

        system_instruction_data = self.system_instruction()
        if system_instruction_data:
            json_data["system_instruction"] = system_instruction_data

        json_data["contents"] = self.history + [{"role": "user", "parts": self.contents()}]

        logger.info("JSON-данные успешно собраны: %s", json_data)
        return json_data
