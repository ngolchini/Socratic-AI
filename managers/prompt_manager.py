from typing import Dict, List, Optional
import json
from pathlib import Path
import logging
from enum import Enum

from models.phase import PhaseType, PhaseConfig
from models.assessment import TopicRelevance, RedirectType, TopicAssessment
from models.phase import PhaseType, ClinicalElement

class GuidanceLevel(Enum):
    HIGH = "high"      # For basic learners - provides explicit guidance
    MEDIUM = "medium"  # For intermediate learners - offers subtle hints
    LOW = "low"        # For advanced learners - minimal guidance

class PromptManager:
    """Manages the construction and maintenance of system prompts for the clinical case discussion."""
    
    def __init__(self, prompts_directory: str = "prompts", default_guidance: GuidanceLevel = GuidanceLevel.MEDIUM):
        self.prompts_dir = Path(prompts_directory)
        self.base_instructions = self._load_base_instructions()
        self.phase_instructions = self._load_phase_instructions()
        self.redirect_templates = self._load_redirect_templates()
        self.logger = logging.getLogger(__name__)
        self.guidance_level = default_guidance
    
    def set_guidance_level(self, level: GuidanceLevel):
        """Set the guidance level for prompts."""
        self.guidance_level = level
        self.logger.info(f"Guidance level set to: {level.value}")

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
    
    def construct_system_prompt(self, phase_context: Dict, guidance_level: Optional[GuidanceLevel] = None) -> str:
        """
        Construct the complete system prompt for the current phase and context.
        Adjusts the instruction style based on the guidance level.
        
        Args:
            phase_context: Context dictionary with phase information
            guidance_level: Optional guidance level override
        """
        # Use provided guidance level or default to current class level
        active_guidance_level = guidance_level or self.guidance_level
        
        phase_type = PhaseType(phase_context["phase_type"])
        phase_json = self.phase_instructions[phase_type]
        phase_config = self.get_phase_config(phase_type)
        
        # Base instructions with guidance level adjustment
        guidance_instructions = self._get_guidance_specific_instructions(active_guidance_level)
        
        prompt_parts = [
            self.base_instructions,
            guidance_instructions,
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
        
        # Add completion block rationale if available
        if "completion_block_rationale" in phase_context:
            completion_block = f"\nNote: Previous attempt to complete this phase was blocked because: {phase_context['completion_block_rationale']}"
            prompt_parts.append(completion_block)
            
        return "\n".join(filter(None, prompt_parts))
    
    def _get_guidance_specific_instructions(self, guidance_level: GuidanceLevel) -> str:
        """Return guidance-level specific instructions."""
        if guidance_level == GuidanceLevel.HIGH:
            return """
            GUIDANCE LEVEL: HIGH
            As a medical educator, provide direct and explicit guidance:
            - Offer clear explanations connecting findings to potential diagnoses
            - Directly suggest diagnoses or test considerations when appropriate
            - Point out missing information or inconsistencies explicitly
            - Provide detailed explanations of pathophysiology and clinical correlations
            - Use a supportive tone while providing structured guidance
            """
        elif guidance_level == GuidanceLevel.MEDIUM:
            return """
            GUIDANCE LEVEL: MEDIUM
            As a medical educator, provide balanced guidance:
            - Ask guiding questions that lead to insights rather than stating them directly
            - Suggest considerations obliquely rather than naming specific diagnoses
            - Point out patterns but let the learner make connections
            - Provide moderate explanations when needed, but encourage independent reasoning
            - Balance support with allowing productive struggle
            """
        else:  # LOW
            return """
            GUIDANCE LEVEL: LOW
            As a medical educator, provide minimal guidance:
            - Avoid leading questions that suggest specific diagnoses
            - Do not volunteer connections between findings unless explicitly asked
            - Provide factual information only when directly requested
            - Let the learner arrive at their own conclusions, even if incomplete
            - Only redirect if the learner is completely off-track
            - Allow productive struggle and discovery
            """

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
        phase_type: PhaseType,
        guidance_level: Optional[GuidanceLevel] = None
    ) -> str:
        """
        Generate an appropriate redirection message based on topic assessment.
        
        Args:
            assessment: TopicAssessment containing relevance and prohibited topics
            phase_type: Current phase type
            guidance_level: Optional guidance level override
        """
        if assessment.relevance == TopicRelevance.ON_TOPIC:
            return ""
        
        # Use provided guidance level or default to current class level
        active_guidance_level = guidance_level or self.guidance_level
        
        # Determine redirection type based on guidance level
        # For high guidance, use direct redirection
        # For medium guidance, use educational redirection
        # For low guidance, use gentle redirection
        if assessment.redirect_type == RedirectType.NONE:
            if active_guidance_level == GuidanceLevel.HIGH:
                redirect_type = RedirectType.DIRECT
            elif active_guidance_level == GuidanceLevel.MEDIUM:
                redirect_type = RedirectType.EDUCATIONAL
            else:  # LOW
                redirect_type = RedirectType.GENTLE
        else:
            redirect_type = assessment.redirect_type
        
        try:
            template = self.redirect_templates[redirect_type]
        except KeyError:
            self.logger.error(f"Redirect type {redirect_type} not found in templates.")
            return "The topic you mentioned is not appropriate for this phase. Please focus on the current phase topics."
        
        # Customize redirection based on phase and prohibited topics
        topic = "that topic"
        appropriate_phase = "later"
        
        if hasattr(assessment, 'prohibited_topics') and assessment.prohibited_topics:
            topic = assessment.prohibited_topics[0]
            appropriate_phase = next(
                (p.value for p in PhaseType if topic in self.phase_instructions[p].get("relevant_topics", [])),
                "later"
            )

        return template.format(
            topic=topic,
            appropriate_phase=appropriate_phase,
            current_phase_focus=self.phase_instructions[phase_type]["focus_area"]
        )

    def generate_phase_transition(
        self,
        current_phase: PhaseType,
        next_phase: PhaseType,
        covered_points: List[str],
        guidance_level: Optional[GuidanceLevel] = None
    ) -> str:
        """
        Generate a phase transition message including summary and next steps.
        
        Args:
            current_phase: Phase being completed
            next_phase: Phase being transitioned to
            covered_points: Teaching points covered in current phase
            guidance_level: Optional guidance level override
        """
        # Use provided guidance level or default to current class level
        active_guidance_level = guidance_level or self.guidance_level
        
        current_phase_info = self.phase_instructions[current_phase]
        next_phase_info = self.phase_instructions[next_phase]
        
        # Adjust transition style based on guidance level
        if active_guidance_level == GuidanceLevel.HIGH:
            # More detailed and instructive transition
            transition_parts = [
                current_phase_info["completion_message"],
                f"\nYou've successfully completed the {current_phase.value.capitalize()} phase! Here's what we learned:",
                self._format_teaching_points(covered_points),
                f"\nNow we'll move to the {next_phase.value.capitalize()} phase, where we'll focus on:",
                next_phase_info["introduction"],
                f"\nRemember to apply what we learned in the {current_phase.value} phase as we proceed."
            ]
        elif active_guidance_level == GuidanceLevel.MEDIUM:
            # Standard transition
            transition_parts = [
                current_phase_info["completion_message"],
                "\nKey points covered:",
                self._format_teaching_points(covered_points),
                "\nNext Phase:",
                next_phase_info["introduction"]
            ]
        else:  # LOW
            # Minimal transition
            transition_parts = [
                current_phase_info["completion_message"],
                "\nMoving to the next phase:",
                next_phase_info["introduction"]
            ]
        
        return "\n".join(filter(None, transition_parts))

    def construct_probe_question(
        self, 
        missing_element: ClinicalElement, 
        phase_context: Dict,
        guidance_level: Optional[GuidanceLevel] = None
    ) -> str:
        """
        Generate a question to probe for specific missing information,
        with style adjusted based on guidance level.
        
        Args:
            missing_element: Clinical element that needs to be covered
            phase_context: Current phase context
            guidance_level: Optional guidance level override
        """
        # Use provided guidance level or default to current class level
        active_guidance_level = guidance_level or self.guidance_level
        
        phase_type = PhaseType(phase_context["phase_type"])
        phase_instructions = self.phase_instructions[phase_type]
        
        # Get appropriate template based on guidance level
        if active_guidance_level == GuidanceLevel.HIGH:
            # Direct probe for high guidance
            probe_template = phase_instructions.get("probe_template_high", 
                "We should also discuss {element}. Can you tell me what you know about this?"
            )
        elif active_guidance_level == GuidanceLevel.MEDIUM:
            # Balanced probe for medium guidance
            probe_template = phase_instructions.get("probe_template_medium", 
                "What about {element}? Is that relevant in this case?"
            )
        else:  # LOW
            # Subtle probe for low guidance
            probe_template = phase_instructions.get("probe_template_low", 
                "What other aspects of the case are you considering?"
            )
        
        # Generate context-aware probe
        if active_guidance_level == GuidanceLevel.LOW:
            # For low guidance, don't directly mention the missing element
            return probe_template
        else:
            return probe_template.format(
                element=missing_element.content.lower()
            )

    def construct_follow_up_question(
        self, 
        phase_context: Dict,
        guidance_level: Optional[GuidanceLevel] = None
    ) -> str:
        """
        Generate an appropriate follow-up question based on phase context.
        
        Args:
            phase_context: Current phase context
            guidance_level: Optional guidance level override
        """
        # Use provided guidance level or default to current class level
        active_guidance_level = guidance_level or self.guidance_level
        
        phase_type = PhaseType(phase_context["phase_type"])
        phase_instructions = self.phase_instructions[phase_type]
        
        # Get remaining elements to cover
        uncovered = set(phase_context["required_elements"]) - set(phase_context["covered_elements"])
        
        if uncovered:
            # Select template based on guidance level
            if active_guidance_level == GuidanceLevel.HIGH:
                # More direct for high guidance
                template = phase_instructions.get("follow_up_template_high", 
                    "Let's continue gathering information. What would you like to know about the patient's {phase} next?"
                )
            elif active_guidance_level == GuidanceLevel.MEDIUM:
                # Balanced for medium guidance
                template = phase_instructions.get("follow_up_template_medium", 
                    "What else would you like to know about the patient's {phase}?"
                )
            else:  # LOW
                # Open-ended for low guidance
                template = phase_instructions.get("follow_up_template_low", 
                    "What are your next steps in approaching this case?"
                )
                
            return template.format(phase=phase_type.value)
        else:
            # All elements covered, time for synthesis
            if active_guidance_level == GuidanceLevel.HIGH:
                # More guided synthesis for high guidance
                template = phase_instructions.get("synthesis_template_high",
                    "Now that we've covered the key {phase} elements, what's your assessment of this patient's condition?"
                )
            elif active_guidance_level == GuidanceLevel.MEDIUM:
                # Balanced synthesis for medium guidance
                template = phase_instructions.get("synthesis_template_medium",
                    "Based on what we've discussed, what are your initial thoughts?"
                )
            else:  # LOW
                # Very open synthesis for low guidance
                template = phase_instructions.get("synthesis_template_low",
                    "What's your current thinking on this case?"
                )
                
            return template.format(phase=phase_type.value)

    def generate_teaching_prompt(
        self,
        teaching_point: str,
        phase_type: PhaseType,
        guidance_level: Optional[GuidanceLevel] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Generate a teaching-focused response to guide learning,
        with the style adjusted based on guidance level.
        
        Args:
            teaching_point: The teaching point to discuss
            phase_type: Current phase type
            guidance_level: Optional guidance level override
            context: Optional context about previous discussion
        """
        # Use provided guidance level or default to current class level
        active_guidance_level = guidance_level or self.guidance_level
        
        phase_instructions = self.phase_instructions[phase_type]
        
        # Adjust template based on guidance level
        if active_guidance_level == GuidanceLevel.HIGH:
            template = phase_instructions.get("teaching_template_high",
                "That's an important point about {point}. {explanation}"
            )
            # For high guidance, provide explanation rather than question
            content = self._teaching_point_to_explanation(teaching_point)
        elif active_guidance_level == GuidanceLevel.MEDIUM:
            template = phase_instructions.get("teaching_template_medium",
                "That's an interesting observation. {question}"
            )
            # For medium guidance, use Socratic question
            content = self._teaching_point_to_question(teaching_point)
        else:  # LOW
            template = phase_instructions.get("teaching_template_low",
                "I see. {minimal_prompt}"
            )
            # For low guidance, use minimal prompt
            content = self._teaching_point_to_minimal_prompt(teaching_point)
        
        # Format with appropriate content based on guidance level
        if active_guidance_level == GuidanceLevel.HIGH:
            return template.format(point=teaching_point.strip().rstrip('.'), explanation=content)
        elif active_guidance_level == GuidanceLevel.MEDIUM:
            return template.format(question=content)
        else:  # LOW
            return template.format(minimal_prompt=content)

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
            
    def _teaching_point_to_explanation(self, teaching_point: str) -> str:
        """Convert a teaching point into a direct explanation (for HIGH guidance)."""
        # Remove any leading/trailing whitespace and periods
        point = teaching_point.strip().rstrip('.')
        
        # Format as direct explanation
        if "relationship between" in point.lower():
            return f"We should understand the {point} because it helps us connect pathophysiology to clinical presentation."
        elif "importance of" in point.lower():
            return f"The {point} is critical because it directly impacts our diagnostic approach and management."
        elif "approach to" in point.lower():
            return f"Developing {point} involves considering the key clinical signs and differential diagnoses we've discussed."
        else:
            return f"This relates to {point}, which is a key concept in understanding this case."
    
    def _teaching_point_to_minimal_prompt(self, teaching_point: str) -> str:
        """Create a minimal, non-leading prompt (for LOW guidance)."""
        # Remove any leading/trailing whitespace and periods
        point = teaching_point.strip().rstrip('.')
        
        # Very subtle prompts that avoid leading the learner
        if "relationship between" in point.lower():
            return "What factors might be relevant to your analysis here?"
        elif "importance of" in point.lower():
            return "What considerations are guiding your thinking at this point?"
        elif "approach to" in point.lower():
            return "How are you approaching this aspect of the case?"
        else:
            return "What's your current thought process?"