from typing import Optional, List, Dict, Tuple, Any
import json
import logging  # Add this import
import streamlit as st
from datetime import datetime

from models.phase import Phase, PhaseType, TeachingPoint, ClinicalElement
from models.case import CaseData
from models.assessment import (
    TopicAssessment,
    CoverageAssessment,
    TopicRelevance,
    RedirectType
)
from managers.llm_manager import LLMManager  # Add this import
from managers.prompt_manager import PromptManager 

class PhaseManager:
    """Manages the progression through clinical case phases."""

class PhaseManager:
    def __init__(self, case_data: CaseData, llm_manager: LLMManager, prompt_manager: PromptManager):
        """Initialize the phase manager."""
        self.case_data = case_data
        self.current_phase_type = st.session_state.current_phase
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager  # Add this line
        self.coverage_cache: Dict[str, bool] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_phase()
        
        # Add new attributes for phase management
        self.current_phase_prompt = None
        self.last_completion_block_rationale = None

    def _initialize_phase(self):
        """Initialize the current phase and its tracking mechanisms."""
        self.current_phase = self.case_data.get_current_phase(self.current_phase_type)
        self.last_completion_block_rationale = None
        
        # Initialize elements 
        for element in self.current_phase.required_elements + self.current_phase.optional_elements:
            if self.coverage_cache.get(element.id, False):
                element.elicited = True
            elif element.elicited is None:
                element.elicited = False
                element.elicited_content = None
        
        # Initialize teaching points
        for point in self.current_phase.teaching_points:
            if point.covered is None:
                point.covered = False
                point.coverage_notes = None

        # Construct the system prompt and store in session state
        phase_context = self.get_phase_context()
        system_prompt = self.prompt_manager.construct_system_prompt(phase_context)
        st.session_state.current_phase_prompt = system_prompt
        self.current_phase_prompt = system_prompt
        
        self.logger.info(f"Initialized phase {self.current_phase_type.value} with prompt: {system_prompt}")

    def assess_user_message(self, message: str) -> Tuple[TopicAssessment, CoverageAssessment]:
        """
        Legacy method maintained for backwards compatibility.
        Delegates to new separate assessment methods.
        """
        topic_assessment = self.assess_topic(message)
        if topic_assessment.relevance != TopicRelevance.OFF_TOPIC:
            coverage_assessment = self.assess_coverage(st.session_state.chat_messages)
        else:
            coverage_assessment = CoverageAssessment([], [], [])
        return topic_assessment, coverage_assessment

    def assess_topic(self, message: str) -> TopicAssessment:
        """
        Evaluate if the user's message is appropriate for clinical case discussion.
        Runs before getting LLM response to ensure appropriate interaction.
        Previously included the line "3. Does it avoid prohibited topics like: {prohibited_topics}"
        """
        system_prompt = """You are a clinical case discussion moderator.
        Assess if this message is appropriate for a clinical educational discussion.
        Consider:
        1. Is it related to clinical medicine or medical education?
        2. Is it respectful and professional?

        
        Return a JSON object with:
        - appropriate (boolean): whether the message is suitable
        - redirect_type (string): "none", "direct", or "gentle"
        - redirect_message (string, optional): suggested redirection if needed
        """

        assessment = self.llm_manager.get_json_response(
            #system_prompt.format(
            #    prohibited_topics=self.current_phase.config.prohibited_topics
            #),
            system_prompt,
            message
        )
        
        try:
            if assessment.get("appropriate", False):
                relevance = TopicRelevance.ON_TOPIC
                redirect_type = RedirectType.NONE
            else:
                relevance = TopicRelevance.OFF_TOPIC
                redirect_type = RedirectType(assessment.get("redirect_type", "direct"))
            
            #prohibited_topics = assessment.get("prohibited_topics", [])
            redirect_message = assessment.get("redirect_message")
            
            self.logger.info(f"Topic assessment: relevance={relevance}, redirect_type={redirect_type}")
            
            return TopicAssessment(
                relevance=relevance,
                redirect_type=redirect_type,
                #prohibited_topics=prohibited_topics,
                redirect_message=redirect_message
            )
        except Exception as e:
            self.logger.error(f"Error in topic assessment: {str(e)}")
            return TopicAssessment(
                relevance=TopicRelevance.OFF_TOPIC,
                redirect_type=RedirectType.DIRECT,
                #prohibited_topics=[],
                redirect_message="I'm sorry, I couldn't properly assess that message. Could you rephrase it?"
            )

    def assess_coverage(self, user_message: str, assistant_response: str) -> CoverageAssessment:
        """
        Analyze the latest message exchange to determine which clinical elements have been covered.
        More efficient than analyzing full chat history for every message.
        """
        self.logger.info("=== Starting Coverage Assessment for Latest Exchange ===")
        
        # Get uncovered elements to check
        elements_to_check = []
        for element in self.current_phase.required_elements + self.current_phase.optional_elements:
            if self.coverage_cache.get(element.id, False):
                element.elicited = True
                continue
            elements_to_check.append(element)
        
        if not elements_to_check:
            return CoverageAssessment([], [], [])
        
        # Just analyze the latest exchange
        exchange_text = f"""USER: {user_message}
    ASSISTANT: {assistant_response}"""
        
        system_prompt = """Assess which clinical elements have been adequately covered in this latest exchange.
        Consider both the question/response pair when determining if an element has been properly addressed. 
        
        Return a JSON object where:
        - Keys are the clinical elements being checked
        - Values are objects containing:
        - covered (boolean): whether the element has been adequately addressed
        - details (string): how it was addressed or what's still missing
        """
        
        user_message = f"Elements to check: {[e.content for e in elements_to_check]}\n\nExchange:\n{exchange_text}"

        response = self.llm_manager.get_json_response(system_prompt, user_message)
        self.logger.info(f"Coverage analysis response: {response}")
        
        newly_covered_elements = []
        for element in elements_to_check:
            element_result = response.get(element.content, {})
            if element_result.get('covered', False):
                self.logger.info(f"Element {element.id} marked as covered")
                newly_covered_elements.append(element)
                self.coverage_cache[element.id] = True
                element.elicited = True
                element.elicited_content = element_result.get('details', '')

        # Update cache for previously covered elements
        for element in (self.current_phase.required_elements + self.current_phase.optional_elements):
            if element.elicited:
                self.coverage_cache[element.id] = True
        
        newly_covered_points = self._update_teaching_points(newly_covered_elements)
        
        missing_critical = [
            element for element in self.current_phase.required_elements
            if not self.coverage_cache.get(element.id, False)
        ]
        
        return CoverageAssessment(
            newly_covered_elements=newly_covered_elements,
            newly_covered_points=newly_covered_points,
            missing_critical_elements=missing_critical
        )

    def _update_teaching_points(self, covered_elements: List[ClinicalElement]) -> List[TeachingPoint]:
        """Update teaching points based on newly covered elements."""
        newly_covered_points = []
        for element in covered_elements:
            for point in element.teaching_points:
                if not point.covered:
                    point.covered = True
                    newly_covered_points.append(point)
        return newly_covered_points

    def check_phase_completion(self, chat_history: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        Check if phase completion criteria are met, with caching.
        """
        # Check if we already know the phase is complete
        if hasattr(st.session_state, 'phase_completion_status'):
            phase_status = st.session_state.phase_completion_status
            if self.current_phase_type.value in phase_status:
                return phase_status[self.current_phase_type.value]

        # Get all required elements for current phase
        required_elements = self.current_phase.required_elements
        
        # Log all required elements first
        self.logger.info("Required elements: %s", [e.content for e in required_elements])
        
        # Quick check of required elements
        for element in required_elements:
            is_elicited = element.elicited or self.coverage_cache.get(element.id, False)
            if not is_elicited:
                self.logger.info(f"Element not covered: {element.content}")
                return False
            # Update element status from cache
            if self.coverage_cache.get(element.id, False):
                element.elicited = True

        # If we get here, all required elements are covered
        # Now do thorough check with full chat history
        self.logger.info("All elements covered, performing thorough completion check...")
        
        # Use provided chat history or get from session state
        if not chat_history and hasattr(st.session_state, 'chat_messages'):
            chat_history = st.session_state.chat_messages
        
        if not chat_history:
            self.logger.warning("No chat history available for completion check")
            return False

        # Only perform the full LLM check if we haven't already determined completion
        if not hasattr(st.session_state, 'phase_completion_status'):
            st.session_state.phase_completion_status = {}

        if self.current_phase_type.value not in st.session_state.phase_completion_status:
            system_prompt = f"""You are assessing if a clinical case discussion in the {self.current_phase_type.value} phase is ready to advance.
            
            Phase advancement criteria: {self.current_phase.config.advancement_criteria}
            
            Evaluate the discussion for whether these criteria have been reasonably met based on the chat history.
            You do not need to insist upon a comprehensive discussion or mastery of every element, but they should be at least touched upon. 
            
            Return a JSON object with:
            - can_advance (boolean): whether the phase can be completed
            - rationale (string): explanation of the decision
            - missing_aspects (list, optional): any aspects still needing attention
            """
            
            # Format chat history for LLM
            formatted_chat = self._format_chat_history(chat_history)
            
            user_message = f"""Covered elements: {[e.content for e in required_elements]}
            Chat history:
            {formatted_chat}"""

            response = self.llm_manager.get_json_response(system_prompt, user_message)
            self.logger.info(f"Phase completion assessment: {response}")
            
            can_advance = response.get("can_advance", False)
            if not can_advance:
                self.last_completion_block_rationale = response.get("rationale")
                self.logger.info(f"Phase completion blocked: {self.last_completion_block_rationale}")
                if response.get("missing_aspects"):
                    self.logger.info(f"Missing aspects: {response['missing_aspects']}")
            else:
                self.last_completion_block_rationale = None
                self.logger.info("Phase completion check passed")
                
            # Cache the result
            st.session_state.phase_completion_status[self.current_phase_type.value] = can_advance
                    
            return can_advance
        
        return st.session_state.phase_completion_status[self.current_phase_type.value]
    
    def _format_chat_history(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history for LLM consumption."""
        if not chat_history:
            return "No substantive discussion has occurred yet."
            
        formatted_messages = []
        for msg in chat_history:
            if msg["role"] in ["user", "assistant"]:
                formatted_messages.append(f"{msg['role'].upper()}: {msg['content']}")
        
        return "\n".join(formatted_messages) if formatted_messages else "No relevant messages found."

    def advance_phase(self) -> Optional[str]:
        """Attempt to advance to the next phase if completion criteria are met."""
        if not self.check_phase_completion():
            return None

        completion_message = self.current_phase.config.completion_message
        
        # Determine next phase
        current_index = list(PhaseType).index(self.current_phase_type)
        if current_index < len(PhaseType) - 1:
            self.current_phase_type = list(PhaseType)[current_index + 1]
            # Update session state with new phase
            st.session_state.current_phase = self.current_phase_type
            # Reload phase config for new phase
            self._initialize_phase()
            self.current_phase = self.case_data.get_current_phase(self.current_phase_type)
        
        return completion_message
        
    def generate_phase_summary(
            self,
            chat_history: List[Dict[str, str]]
        ) -> Dict[str, Any]:
        """
        Generate comprehensive summaries of the phase using chat history.
        
        Args:
            chat_history: List of chat messages to analyze
        
        Returns:
            Dictionary containing:
            - chat_summary: Detailed summary for chat display
            - learner_assessment: Private assessment of learner performance
            - clinical_summary: Structured summary of clinical findings
        """
        self.logger.info(f"Generating phase summary for {self.current_phase_type.value}")
        
        # Format chat history for LLM consumption
        formatted_chat = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in chat_history
            if msg['role'] in ['user', 'assistant']
        ])
        
        system_prompt = f"""You are analyzing a clinical case discussion in the {self.current_phase_type.value} phase.
        Generate three distinct summaries of the conversation:

        1. A chat summary that:
        - Highlights key points discussed
        - Notes important clinical reasoning demonstrated
        - Identifies key teaching points covered
        - Provides a natural transition to the next phase
        
        2. A learner assessment that evaluates:
        - Thoroughness of history/examination approach
        - Clinical reasoning quality
        - Knowledge gaps identified
        - Areas for improvement
        - Particular strengths shown
        
        3. A clinical summary that structures:
        - Key findings and their significance
        - Pertinent negatives
        - Risk factors identified
        - Working differential diagnosis
        
        Return a JSON object with these three summaries as separate fields.
        For the clinical summary, structure the findings by category (e.g., symptoms, risk factors, exam findings).
        """
        
        response = self.llm_manager.get_json_response(
            system_prompt,
            f"Phase: {self.current_phase_type.value}\n\nConversation:\n{formatted_chat}"
        )
        
        try:
            chat_summary = response.get("chat_summary", "Phase summary could not be generated.")
            learner_assessment = response.get("learner_assessment", {})
            clinical_summary = response.get("clinical_summary", {})
            
            # Store learner assessment for potential future use
            if 'learner_assessments' not in st.session_state:
                st.session_state.learner_assessments = {}
            st.session_state.learner_assessments[self.current_phase_type.value] = learner_assessment
            
            # Prepare the phase summary for storage
            phase_summary = {
                "phase": self.current_phase_type.value,
                "completion_time": datetime.now(),
                "findings_summary": clinical_summary,
                "covered_elements": [e.content for e in self.current_phase.required_elements if e.elicited],
                "learner_assessment": learner_assessment,
            }
            
            return {
                "chat_summary": chat_summary,
                "learner_assessment": learner_assessment,
                "clinical_summary": clinical_summary,
                "phase_summary": phase_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error processing phase summary: {str(e)}")
            return {
                "chat_summary": "Error generating phase summary.",
                "learner_assessment": {},
                "clinical_summary": {},
                "phase_summary": {}
            }
        
    def get_phase_context(self) -> Dict[str, any]:
        """
        Get the current phase context for prompt construction.
        TODO: Add the most recent rationale for phase completion failure.
        """
        return {
            "phase_type": self.current_phase_type.value,
            "required_elements": [e.content for e in self.current_phase.required_elements],
            "covered_elements": [e.content for e in self.current_phase.required_elements if e.elicited],
            "teaching_points": [p.content for p in self.current_phase.teaching_points if not p.covered],
            "prohibited_topics": self.current_phase.config.prohibited_topics
        }