from openai import OpenAI
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

class LLMManager:
    """Manages LLM interactions with distinct handling for conversational and one-off requests."""
    
    def __init__(self, client: OpenAI, default_guidance_level="medium"):
        self.client = client
        self.model = "gpt-4.1"
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
            
            # Extract structured data if present
            try:
                context = json.loads(user_message)
                patient_record = context.get("patient_record", {})
                user_text = context.get("user_text", "")
            except Exception:
                patient_record = {}
                user_text = user_message

            # Format record - but make it educational, not restrictive
            record_text = []
            if patient_record:
                record_text.append("# AVAILABLE PATIENT INFORMATION")
                
                for section, items in patient_record.items():
                    if isinstance(items, list) and items:
                        record_text.append(f"\n## {section.replace('_', ' ').title()}:")
                        record_text.extend(f"- {item}" for item in items if isinstance(item, str))
                    elif isinstance(items, dict) and items:
                        record_text.append(f"\n## {section.replace('_', ' ').title()}:")
                        record_text.extend(f"- {k}: {v}" for k, v in items.items())

                formatted_record = "\n".join(record_text)
                
                # More educational approach - guide rather than restrict
                messages.append({
                    "role": "user",
                    "content": f"""{formatted_record}

    IMPORTANT EDUCATIONAL GUIDELINES:
    - Continue being a Socratic tutor as defined in your system prompt. Use the case reference material above to: 
    - Ground your responses in actual case facts (don't make up findings) 
    - When learners ask about specific findings, use the information above if available 
    - If they ask about something not in the reference material, guide them back to what IS available through questioning 
    - Maintain your educational approach 
    - ask questions, probe reasoning, provide hints based on guidance level 
    - Don't just give information 
    - teach through questioning and guided discovery

    USER QUESTION: {user_text}
    """
                })
            else:
                # No patient record available, proceed normally
                messages.append({"role": "user", "content": user_text})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"LLM error in conversational response: {str(e)}")
            return None
        

    def _get_guidance_instruction_prefix(self) -> str:
        """Get a short prefix indicating guidance level."""
        return f"GUIDANCE LEVEL: {self.guidance_level.upper()}"

    def _get_guidance_instructions(self) -> str:
        """Get guidance-specific instructions based on the current guidance level."""
        base_instruction = """
        CORE TUTORING PRINCIPLES:
        - Use Socratic method - guide through targeted questions, not just listing information
        - When a broad topic is mentioned (like PMH), ALWAYS probe for specifics
        - Break down complex medical histories into individual elements to explore
        - Ask follow-up questions about timing, severity, treatment, and current status
        - Don't just confirm information exists - help learner extract clinical significance
        
        WHEN LEARNER ASKS ABOUT PMH:
        - Don't just list conditions, never reveal parts of the case that have not been explicity requested
        - Promptthe learner to ask specific questions about each condition: "Tell me about the schizophrenia - when was it diagnosed?"
        - Probe for details: "What about the substance abuse history - what substances and when?"
        - Guide toward clinical relevance: "How might this affect our current assessment?"
        """
        
        if self.guidance_level == "low":
            return base_instruction + """
            LOW GUIDANCE - Minimal Probing:
            - Ask one open-ended follow-up: "What specific details about the PMH are most relevant?"
            - Let learner choose which elements to explore
            - Use general prompts: "What else would you like to know about these conditions?"
            """
        elif self.guidance_level == "medium":
            return base_instruction + """
            MEDIUM GUIDANCE - Focused Probing:
            - Ask about 2-3 specific PMH elements: "Tell me about the substance abuse history"
            - Guide attention to clinically relevant items: "The schizophrenia diagnosis - how might that be relevant here?"
            - Suggest areas to explore: "Which of these conditions might impact the current presentation?"
            """
        else:  # high guidance
            return base_instruction + """
            HIGH GUIDANCE - Detailed Probing:
            - Ask specific questions about each PMH element systematically
            - Provide clear direction: "Let's explore each of these - starting with the schizophrenia, when was it diagnosed?"
            - Explain why details matter: "The timing of the substance abuse is important because..."
            - Guide through each element methodically
            """









