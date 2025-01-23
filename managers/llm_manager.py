from openai import OpenAI
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class LLMManager:
    """Manages LLM interactions with distinct handling for conversational and one-off requests."""
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = "gpt-4o"
        self.logger = logging.getLogger(__name__)

    def _prepare_message_history(self, message_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert message history to format expected by OpenAI API."""
        prepared_messages = []
        for msg in message_history:
            if msg["role"] in ["user", "assistant"]:
                prepared_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        return prepared_messages

    def get_json_response(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Get a structured JSON response from the LLM (single exchange)."""
        try:
            messages = [
                {"role": "system", "content": f"{system_prompt}\nPlease respond in JSON format."},
                {"role": "user", "content": user_message}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            self.logger.info("Single exchange - System prompt: %s", system_prompt)
            self.logger.info("Single exchange - User message: %s", user_message)
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"LLM error in JSON response: {str(e)}")
            return {"error": str(e)}

    def get_text_response(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Get a free-text response from the LLM (single exchange)."""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM error in text response: {str(e)}")
            return None

    def get_conversational_response(
        self,
        system_prompt: str,
        user_message: str,
        message_history: List[Dict[str, Any]],
        temperature: float = 0.7
    ) -> Optional[str]:
        """Get a response maintaining conversation context."""
        try:
            # Start with system prompt
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add prepared conversation history
            messages.extend(self._prepare_message_history(message_history))
            
            # Add current user message with structured format
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            self.logger.info("Conversation - History length: %d", len(message_history))
            self.logger.info("Conversation - Latest message: %s", user_message)
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM error in conversational response: {str(e)}")
            return None
