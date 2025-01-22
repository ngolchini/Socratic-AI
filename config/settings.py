# config/settings.py
from pathlib import Path
from typing import Dict, Any
import json
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class LLMConfig:
    """Configuration for LLM interactions."""
    model_name: str
    temperature: float
    max_tokens: int
    frequency_penalty: float
    presence_penalty: float

@dataclass
class PhaseConfig:
    """Configuration for phase management."""
    evidence_threshold: float
    required_coverage_percentage: float
    max_redirections: int
    advancement_delay: float

@dataclass
class UIConfig:
    """Configuration for user interface."""
    chat_column_ratio: float
    max_differential_items: int
    debug_mode_default: bool
    animation_speed: float

@dataclass
class LogConfig:
    """Configuration for logging."""
    log_level: str
    log_format: str
    log_file: Path
    enable_console: bool

@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig
    phase: PhaseConfig
    ui: UIConfig
    log: LogConfig
    cases_directory: Path
    prompts_directory: Path

class Settings:
    """Global settings management."""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path("config/config.json")
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from JSON file."""
        with open(self.config_path) as f:
            config_data = json.load(f)
            
        return AppConfig(
            llm=LLMConfig(**config_data["llm"]),
            phase=PhaseConfig(**config_data["phase"]),
            ui=UIConfig(**config_data["ui"]),
            log=LogConfig(
                **config_data["log"],
                log_file=Path(config_data["log"]["log_file"])
            ),
            cases_directory=Path(config_data["cases_directory"]),
            prompts_directory=Path(config_data["prompts_directory"])
        )
    
    @lru_cache()
    def get_settings(self) -> AppConfig:
        """Get cached configuration settings."""
        return self.config