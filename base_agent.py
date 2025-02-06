import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List
from pydantic import BaseModel, ConfigDict, PrivateAttr

class BaseAgent(BaseModel, ABC):
    """
    Базовый класс для всех агентов.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "BaseAgent"
    _logger: logging.Logger = PrivateAttr()  # Теперь логгер создаётся в `__init__()`

    def __init__(self, **data):
        """Инициализация логгера"""
        super().__init__(**data)
        self._setup_logger()

    def _setup_logger(self):
        """Настраивает логгер"""
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "agents.log")

        # Get logger for this agent instance
        self._logger = logging.getLogger(f"{self.name}")
        self._logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels
        
        # Remove existing handlers to prevent duplicates
        if self._logger.hasHandlers():
            self._logger.handlers.clear()
        
        # Create both file and stream handlers
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        stream_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Set formatter for both handlers
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        # Set levels for handlers
        file_handler.setLevel(logging.DEBUG)
        stream_handler.setLevel(logging.INFO)
        
        # Add handlers to logger
        self._logger.addHandler(file_handler)
        self._logger.addHandler(stream_handler)
        
        # Ensure propagation is False to prevent duplicate logs
        self._logger.propagate = False
        
        # Test log message
        self._logger.info(f"Logger initialized for {self.name}")

    def log(self, message: str, level: str = "INFO") -> None:
        """Логирует сообщение."""
        level = level.upper()  # Normalize level to uppercase
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message)
        
        # Force flush handlers
        for handler in self._logger.handlers:
            handler.flush()

    def handle_error(self, error: Exception, context: Any = None) -> None:
        """Обрабатывает ошибки."""
        self.log(f"Error: {error}. Context: {context}", level="ERROR")

    def validate_data(self, data: List[Any]) -> bool:
        """Проверяет входные данные перед обработкой."""
        if not isinstance(data, list) or not data:
            self.log("Invalid or empty data provided.", level="WARNING")
            return False
        return True

    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Метод обработки данных (должен быть реализован в наследниках)."""
        pass
