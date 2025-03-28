from openai import OpenAI
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class LLMManager:
    """Manages LLM interactions with distinct handling for conversational and one-off requests."""
    
    def __init__(self, client: OpenAI, default_guidance_level="medium"):
        self.client = client
        self.model = "gpt-4o"
        self.logger = logging.getLogger(__name__)
        self.guidance_level = default_guidance_level
        
    def set_guidance_level(self, level: str):
        """Set the guidance level for LLM interactions."""
        self.guidance_level = level
        self.logger.info(f"LLM guidance level set to: {level}")

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
            # Add guidance level information to system prompt
            guidance_specific = self._get_guidance_instruction_prefix()
            enhanced_system_prompt = f"{guidance_specific}\n\n{system_prompt}\nPlease respond in JSON format."
            
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
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
            # Add guidance level information to system prompt
            guidance_specific = self._get_guidance_instruction_prefix()
            enhanced_system_prompt = f"{guidance_specific}\n\n{system_prompt}"
            
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
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
        """Get a response maintaining conversation context with guidance level awareness."""
        try:
            # Add guidance-specific instructions to system prompt
            guidance_instructions = self._get_guidance_instructions()
            enhanced_system_prompt = f"{system_prompt}\n\n{guidance_instructions}"
            
            # Start with system prompt
            messages = [{"role": "system", "content": enhanced_system_prompt}]
            
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

    def _get_guidance_instruction_prefix(self) -> str:
        """Get a short prefix indicating guidance level."""
        return f"GUIDANCE LEVEL: {self.guidance_level.upper()}"

    def _get_guidance_instructions(self) -> str:
        """Get guidance-specific instructions based on the current guidance level."""
        if self.guidance_level == "low":
            return """
            RESPONSE GUIDELINES:
            - Avoid leading questions that suggest specific diagnoses
            - Let the learner reach their own conclusions, even if incorrect
            - Do not volunteer connections between findings and diagnoses
            - Only provide information that is explicitly requested
            - Do not suggest next steps or further investigations
            - Respond primarily with neutral factual information
            - Only redirect if the learner is completely off-track
            """
        elif self.guidance_level == "medium":
            return """
            RESPONSE GUIDELINES:
            - Ask open-ended questions that encourage clinical reasoning
            - Provide subtle hints only when the learner is substantially off-track
            - Highlight findings but let the learner make connections themselves
            - Answer questions directly but avoid suggesting specific diagnoses
            - Encourage the learner to synthesize information themselves
            """
        else:  # high guidance
            return """
            RESPONSE GUIDELINES:
            - Provide direct guidance on the clinical reasoning process
            - Explicitly connect findings to potential diagnoses
            - Suggest specific areas the learner should focus on
            - Offer explanations of pathophysiology and diagnostic reasoning
            - Directly identify missed elements and their significance
            """
