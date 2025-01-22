import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

class Logger:
    """Centralized logging configuration."""
    
    def __init__(self, config: LogConfig):
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on settings."""
        # Create logs directory if it doesn't exist
        log_dir = self.config.log_file.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up root logger
        logger = logging.getLogger()
        logger.setLevel(self.config.log_level)
        
        # Create formatters
        file_formatter = logging.Formatter(self.config.log_format)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        # Set up file handler
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Optionally set up console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
    
    @staticmethod
    def log_phase_transition(
        from_phase: str,
        to_phase: str,
        coverage: float,
        duration: float
    ):
        """Log phase transition events."""
        logging.info(
            f"Phase Transition: {from_phase} -> {to_phase} "
            f"(Coverage: {coverage:.1f}%, Duration: {duration:.1f}s)"
        )
    
    @staticmethod
    def log_diagnostic_update(
        diagnosis: str,
        likelihood: float,
        evidence: str
    ):
        """Log updates to differential diagnosis."""
        logging.info(
            f"Diagnostic Update: {diagnosis} "
            f"(Likelihood: {likelihood:.2f}, Evidence: {evidence})"
        )
    
    @staticmethod
    def log_teaching_point(
        point_id: str,
        content: str,
        elicited: bool
    ):
        """Log teaching point coverage."""
        status = "elicited" if elicited else "presented"
        logging.info(f"Teaching Point {point_id} {status}: {content}")
    
    @staticmethod
    def log_user_interaction(
        message_type: str,
        content: str,
        assessment: Optional[Dict] = None
    ):
        """Log user interactions and assessments."""
        logging.info(f"User {message_type}: {content}")
        if assessment:
            logging.debug(f"Assessment: {assessment}")

# config/config.json
{
    "llm": {
        "model_name": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 500,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    },
    "phase": {
        "evidence_threshold": 0.3,
        "required_coverage_percentage": 80.0,
        "max_redirections": 3,
        "advancement_delay": 1.0
    },
    "ui": {
        "chat_column_ratio": 0.6,
        "max_differential_items": 10,
        "debug_mode_default": false,
        "animation_speed": 0.2
    },
    "log": {
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_file": "logs/clinical_tutor.log",
        "enable_console": true
    },
    "cases_directory": "cases",
    "prompts_directory": "prompts"
}