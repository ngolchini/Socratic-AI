from typing import Dict, List, Optional
import json
from pathlib import Path
import logging  # Add this import

from models.phase import PhaseType, PhaseConfig
from models.assessment import TopicRelevance, RedirectType, TopicAssessment
from models.phase import PhaseType, ClinicalElement

class PromptManager:
    """Manages the construction and maintenance of system prompts for the clinical case discussion."""
    
    def __init__(self, prompts_directory: str = "prompts"):
        self.prompts_dir = Path(prompts_directory)
        self.base_instructions = self._load_base_instructions()
        self.phase_instructions = self._load_phase_instructions()
        self.redirect_templates = self._load_redirect_templates()
        self.logger = logging.getLogger(__name__)  # Add this line
        
    def _load_base_instructions(self) -> str:
        """Load the base Socratic teaching instructions."""
        with open(self.prompts_dir / "base.json", "r") as f:
            instructions = json.load(f)
        return instructions["base_instruction"]

    def _load_phase_instructions(self) -> Dict[PhaseType, Dict]:
        """Load phase-specific instruction templates."""
        phase_instructions = {}
        for phase_type in PhaseType:
            with open(self.prompts_dir / "phases" / f"{phase_type.value}.json", "r") as f:
                phase_instructions[phase_type] = json.load(f)
        return phase_instructions

    def _load_redirect_templates(self) -> Dict[RedirectType, str]:
        """Load templates for different types of topic redirections."""
        with open(self.prompts_dir / "redirects.json", "r") as f:
            return json.load(f)

    def get_phase_config(self, phase_type: PhaseType) -> PhaseConfig:
        """Get phase configuration from loaded instructions."""
        phase_json = self.phase_instructions[phase_type]
        return PhaseConfig(
            opening_prompt=phase_json.get("opening_prompt", "What would you like to know about the patient?"),
            completion_message=phase_json.get("completion_message", "Phase complete."),
            prohibited_topics=phase_json.get("prohibited_topics", []),
            advancement_criteria=phase_json.get("advancement_criteria", [])
        )
    
    def construct_system_prompt(self, phase_context: Dict) -> str:
        """
        Construct the complete system prompt for the current phase and context.
        
        Args:
            phase_context: Context dictionary from PhaseManager including:
                - phase_type: Current phase
                - required_elements: List of required clinical elements
                - covered_elements: List of covered elements
                - teaching_points: Remaining teaching points
                - prohibited_topics: Topics to avoid
        """
        phase_type = PhaseType(phase_context["phase_type"])
        phase_json = self.phase_instructions[phase_type]
        phase_config = self.get_phase_config(phase_type)
        
        prompt_parts = [
            self.base_instructions,
            f"\nCurrent Phase: {phase_type.value.capitalize()}\n",
            phase_json["core_instruction"],
            "\nRequired Information to Elicit:",
            self._format_required_elements(
                phase_context["required_elements"],
                phase_context["covered_elements"]
            ),
            "\nTeaching Points to Cover:",
            self._format_teaching_points(phase_context["teaching_points"]),
            "\nTopics to Redirect:",
            self._format_prohibited_topics(phase_config.prohibited_topics),
            phase_json.get("phase_specific_guidance", "")
        ]
        
        return "\n".join(filter(None, prompt_parts))

    def _format_required_elements(
        self,
        required_elements: List[str],
        covered_elements: List[str]
    ) -> str:
        """Format the list of required elements with coverage status."""
        formatted_elements = []
        for element in required_elements:
            status = "✓" if element in covered_elements else "○"
            formatted_elements.append(f"{status} {element}")
        return "\n".join(formatted_elements)

    def _format_teaching_points(self, teaching_points: List[str]) -> str:
        """Format remaining teaching points to be covered."""
        return "\n".join(f"• {point}" for point in teaching_points)

    def _format_prohibited_topics(self, prohibited_topics: List[str]) -> str:
        """Format the list of topics to redirect."""
        return "\n".join(f"- {topic}" for topic in prohibited_topics)

    def generate_redirection(
        self,
        assessment: TopicAssessment,
        phase_type: PhaseType
    ) -> str:
        """
        Generate an appropriate redirection message based on topic assessment.
        
        Args:
            assessment: TopicAssessment containing relevance and prohibited topics
            phase_type: Current phase type
        """
        if assessment.relevance == TopicRelevance.ON_TOPIC:
            return ""
        
        try:
            template = self.redirect_templates[assessment.redirect_type]
        except KeyError:
            self.logger.error(f"Redirect type {assessment.redirect_type} not found in templates.")
            return "The topic you mentioned is not appropriate for this phase. Please focus on the current phase topics."
        
        # Customize redirection based on phase and prohibited topics
        if assessment.prohibited_topics:
            topic = assessment.prohibited_topics[0]
            appropriate_phase = next(
                (p.value for p in PhaseType if topic in self.phase_instructions[p].get("relevant_topics", [])),
                "later"
            )
        else:
            topic = "that topic"
            appropriate_phase = "later"

        return template.format(
            topic=topic,
            appropriate_phase=appropriate_phase,
            current_phase_focus=self.phase_instructions[phase_type]["focus_area"]
        )

    def generate_phase_transition(
        self,
        current_phase: PhaseType,
        next_phase: PhaseType,
        covered_points: List[str]
    ) -> str:
        """
        Generate a phase transition message including summary and next steps.
        
        Args:
            current_phase: Phase being completed
            next_phase: Phase being transitioned to
            covered_points: Teaching points covered in current phase
        """
        current_phase_info = self.phase_instructions[current_phase]
        next_phase_info = self.phase_instructions[next_phase]
        
        transition_parts = [
            current_phase_info["completion_message"],
            "\nKey points covered:",
            self._format_teaching_points(covered_points),
            "\nNext Phase:",
            next_phase_info["introduction"]
        ]
        
        return "\n".join(filter(None, transition_parts))

    def construct_probe_question(self, missing_element: ClinicalElement, phase_context: Dict) -> str:
        """Generate a question to probe for specific missing information."""
        phase_type = PhaseType(phase_context["phase_type"])
        phase_instructions = self.phase_instructions[phase_type]
        
        # Get probe template from phase instructions
        probe_template = phase_instructions.get("probe_template", 
            "Can you tell me more about {element}?"
        )
        
        # Generate context-aware probe
        return probe_template.format(
            element=missing_element.content.lower()
        )

    def construct_follow_up_question(self, phase_context: Dict) -> str:
        """Generate an appropriate follow-up question based on phase context."""
        phase_type = PhaseType(phase_context["phase_type"])
        phase_instructions = self.phase_instructions[phase_type]
        
        # Get remaining elements to cover
        uncovered = set(phase_context["required_elements"]) - set(phase_context["covered_elements"])
        
        if uncovered:
            # Generate question about uncovered area
            return phase_instructions.get("follow_up_template", 
                "What else would you like to know about the patient's history?"
            )
        else:
            # Generate synthesis question
            return phase_instructions.get("synthesis_template",
                "Based on what we've discussed, what are your initial thoughts?"
            )

    def generate_teaching_prompt(
        self,
        teaching_point: str,
        phase_type: PhaseType,
        context: Optional[str] = None
    ) -> str:
        """Generate a teaching-focused response to guide learning."""
        phase_instructions = self.phase_instructions[phase_type]
        
        # Format teaching point as a Socratic question
        template = phase_instructions.get("teaching_template",
            "That's an important observation. {question}"
        )
        
        # Convert teaching point into a question
        question = self._teaching_point_to_question(teaching_point)
        
        return template.format(question=question)

    def _teaching_point_to_question(self, teaching_point: str) -> str:
        """Convert a teaching point into a Socratic question."""
        # Remove any leading/trailing whitespace and periods
        point = teaching_point.strip().rstrip('.')
        
        # Common question starters based on the teaching point content
        if "relationship between" in point.lower():
            return f"How would you explain the {point}?"
        elif "importance of" in point.lower():
            return f"Why do you think {point}?"
        elif "approach to" in point.lower():
            return f"How would you develop {point}?"
        else:
            return f"Can you explain why {point}?"


    def generate_teaching_prompt(
        self,
        teaching_point: str,
        phase_type: PhaseType,
        context: Optional[str] = None
    ) -> str:
        """
        Generate a Socratic prompt to guide discussion of a teaching point.
        
        Args:
            teaching_point: The teaching point to discuss
            phase_type: Current phase type
            context: Optional context about previous discussion
        """
        phase_instructions = self.phase_instructions[phase_type]
        
        prompt_parts = [
            phase_instructions["teaching_prompt_template"],
            f"\nTeaching Point: {teaching_point}"
        ]
        
        if context:
            prompt_parts.append(f"\nContext: {context}")
            
        prompt_parts.append(phase_instructions["teaching_guidance"])
        
        return "\n".join(prompt_parts)