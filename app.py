import streamlit as st
from pathlib import Path
from typing import Optional
import os
import logging
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv
import json

from managers.case_manager import CaseManager
from managers.differential_manager import DifferentialManager
from managers.display_manager import DisplayManager
from managers.phase_manager import PhaseManager
from managers.prompt_manager import PromptManager
from managers.llm_manager import LLMManager 

from models.phase import PhaseType
from models.assessment import TopicAssessment, CoverageAssessment, TopicRelevance
from models.phase import PhaseType


st.set_page_config(
    page_title="Clinical Case Tutor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add after set_page_config
st.components.v1.html(
    """
    <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
    <script src="static/js/differential-helper.js"></script>
    """,
    height=0
)

@dataclass
class LogConfig:
    """Configuration for application logging."""
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    level: int = logging.INFO

def setup_logging(config: LogConfig) -> None:
    """Set up application logging with the specified configuration."""
    logging.basicConfig(
        format=config.format,
        level=config.level
    )

class ClinicalCaseTutor:
    """Main application class for the Clinical Case Tutor system."""
    
    def __init__(self):
        """Initialize the tutor system and its components."""
        # Set up logging
        setup_logging(LogConfig())
        self.logger = logging.getLogger(__name__)

        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client and LLM manager
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.stop()
        
        self.client = OpenAI(api_key=api_key)
        self.llm_manager = LLMManager(self.client)
    
        # Initialize managers in correct order
        self.prompt_manager = PromptManager()
        self.display_manager = DisplayManager(skip_page_config=True)
        self.display_manager.phase_transition_handler = self._handle_phase_transition
        self.case_manager = CaseManager()
        
        # Initialize phase manager and differential manager as None
        self.phase_manager = None
        self.differential_manager = None
        
        # Move session state initialization to constructor
        self._ensure_session_state()
        
        # Initialize managers immediately
        self._initialize_managers()
        
        # Perform initial setup
        self._setup_case()
    
    def _ensure_session_state(self):
        """Ensure all required session state variables exist."""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_case_id = None
            st.session_state.case_loaded = False
            st.session_state.case_data = None
            st.session_state.chat_messages = []
            st.session_state.assessment_cache = {}
            st.session_state.phase_summaries = {}
            st.session_state.differential_diagnosis = []
            st.session_state.current_phase = PhaseType.HISTORY
            st.session_state.differential_manager = None  # Add this line

    def _initialize_managers(self):
        """Initialize all managers with current session state."""
        if st.session_state.case_data:
            self.phase_manager = PhaseManager(
                case_data=st.session_state.case_data, 
                llm_manager=self.llm_manager,
                prompt_manager=self.prompt_manager  # Add this line
            )
            
            # Only create a new DifferentialManager if one doesn't exist
            if st.session_state.differential_manager is None:
                self.differential_manager = DifferentialManager(self.llm_manager)
                st.session_state.differential_manager = self.differential_manager
            else:
                self.differential_manager = st.session_state.differential_manager
        else:
            self.phase_manager = None
            self.differential_manager = None
            st.session_state.differential_manager = None
    
    def _setup_case(self):
        """Set up or continue the current clinical case."""
        available_cases = self._get_available_cases()
        
        with st.sidebar:
            selected_case = st.selectbox(
                "Select Clinical Case",
                options=available_cases,
                index=0 if not st.session_state.current_case_id else 
                available_cases.index(st.session_state.current_case_id),
                key="case_selector"  # Add unique key here
            )
            
            if selected_case != st.session_state.current_case_id:
                st.session_state.case_loaded = False
                self._load_new_case(selected_case)
    
    def _get_available_cases(self) -> list:
        """Get list of available clinical cases."""
        cases_dir = Path("cases")
        return [f.stem for f in cases_dir.glob("*.json")]
    
    def _load_new_case(self, case_id: str):
        """Load a new clinical case and initialize its managers."""
        try:
            self.logger.info(f"Loading case: {case_id}")
            case_data = self.case_manager.load_case(case_id)
            self.logger.info(f"Case loaded successfully: {case_id}")
            
            # Only reset everything if it's a new case
            if st.session_state.current_case_id != case_id:
                st.session_state.chat_messages = []
                st.session_state.assessment_cache = {}
                st.session_state.phase_summaries = {}
                st.session_state.differential_manager = None  # Reset differential manager for new case
            
            # Update session state
            st.session_state.case_data = case_data
            st.session_state.current_case_id = case_id
            st.session_state.case_loaded = True
            st.session_state.current_phase = PhaseType.HISTORY
            
            # Reinitialize managers
            self._initialize_managers()
            
            # Update the initial display before rerunning
            self._update_initial_display()
            
            # Use st.rerun() instead of experimental_rerun
            st.rerun()
            
        except Exception as e:
            self.logger.error(f"Error loading case: {str(e)}")
            st.error(f"Error loading case: {str(e)}")
    
    def _update_initial_display(self):
        """Initialize all display components with current case state."""
        if st.session_state.case_data:
            self.logger.info("Updating initial display with case data")
            self.display_manager.display_case_header(st.session_state.case_data)
            self._update_displays()
            self._display_initial_prompt()
        else:
            self.logger.error("No case data found in session state")
    
    def _update_displays(self):
        """Update all display components with current case state."""
        if not st.session_state.case_loaded or not self.phase_manager:
            return

        try:
            self.logger.info("Updating displays with current case state")
            
            current_phase = self.phase_manager.current_phase_type
            completed_phases = [
                phase_type for phase_type in PhaseType 
                if phase_type in st.session_state.phase_summaries
            ]
            
            # Update phase progress
            self.display_manager.update_phase_progress(
                current_phase=current_phase,
                completed_phases=completed_phases
            )
            
            # Update case information
            if st.session_state.phase_summaries:
                self.display_manager.update_case_information(
                    st.session_state.phase_summaries,
                    current_phase
                )
            
            # Only update differential if we haven't already in this render cycle
            if self.differential_manager and not hasattr(st.session_state, '_differential_updated'):
                self.logger.info("Updating differential display")
                self.display_manager.update_differential_panel(
                    differential_manager=self.differential_manager
                )
                st.session_state._differential_updated = True

        except Exception as e:
            self.logger.error(f"Error updating displays: {str(e)}")
            st.error("An error occurred while updating the display. Please try again.")
            
    def _display_initial_prompt(self):
        """Display the opening prompt for the current phase."""
        if not st.session_state.case_loaded or not self.phase_manager:
            return

        try:
            self.logger.info("Displaying initial prompt for the current phase")
            case_data = st.session_state.case_data
            
            # Initialize chat history if needed
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []
            
            # Show case presentation first
            if not st.session_state.get("case_presented", False):
                # Just use the title as it contains the presentation
                presentation = case_data.metadata.title
                self.logger.info(f"Showing case presentation: {presentation}")
                
                # Add presentation to chat history
                self.display_manager.update_chat_display(
                    message=presentation,
                    role="assistant"
                )
                st.session_state.case_presented = True
            
                # Show the opening prompt
                if hasattr(self.phase_manager.current_phase, 'config'):
                    opening_prompt = self.phase_manager.current_phase.config.opening_prompt
                    self.logger.info(f"Showing opening prompt: {opening_prompt}")
                    
                    # Add opening prompt to chat history
                    self.display_manager.update_chat_display(
                        message=opening_prompt,
                        role="assistant"
                    )
            
            # Force a rerun to ensure display updates
            st.rerun()
                
        except Exception as e:
            self.logger.error(f"Error displaying initial prompt: {str(e)}")
            st.error("An error occurred while starting the case. Please try again.")
                
        except Exception as e:
            self.logger.error(f"Error displaying initial prompt: {str(e)}")
            st.error("An error occurred while starting the case. Please try again.")

    def _display_phase_completion_message(self):
        """Display message that phase criteria are met and show summary button."""
        self.display_manager.update_chat_display(
            "✨ You've covered all the key information for this phase! You can continue the discussion, "
            "or click the button below when you're ready to summarize what we've learned.",
            role="assistant"
        )
        
        # Add button for proceeding to summary
        with self.display_manager.chat_col:
            if st.button("Generate Phase Summary", key="generate_summary"):
                self._generate_phase_summary()
                st.rerun()

    def _generate_phase_summary(self):
        """Generate and display the phase summary."""
        self.logger.info("Generating phase summary...")
        current_phase = self.phase_manager.current_phase_type
        
        # Generate comprehensive summary using chat history
        summaries = self.phase_manager.generate_phase_summary(
            st.session_state.chat_messages
        )
        
        # Store in session state for sidebar display
        if 'phase_summaries' not in st.session_state:
            st.session_state.phase_summaries = {}
        st.session_state.phase_summaries[current_phase] = summaries["phase_summary"]
        
        # Display the chat summary
        self.display_manager.update_chat_display(
            summaries["chat_summary"],
            role="assistant"
        )
        
        # Update the case information display with clinical summary
        self.display_manager.update_case_information(
            st.session_state.phase_summaries,
            current_phase
        )
        
        # Get next phase
        phase_sequence = [
            PhaseType.HISTORY,
            PhaseType.PHYSICAL,
            PhaseType.TESTING,
            PhaseType.MANAGEMENT,
            PhaseType.DISCUSSION
        ]
        
        try:
            current_idx = phase_sequence.index(current_phase)
            next_phase = phase_sequence[current_idx + 1] if current_idx + 1 < len(phase_sequence) else None
        except (ValueError, IndexError):
            self.logger.error(f"Invalid phase transition from {current_phase}")
            next_phase = None
        
        if next_phase:
            st.session_state.pending_next_phase = next_phase
            # Mark that summary has been generated
            st.session_state.summary_generated = True

    
    def _handle_user_input(self, user_input: str):
        """Process and respond to user input using conversational context."""
        if not st.session_state.case_loaded or not self.phase_manager:
            st.error("Please select a case before continuing.")
            return
                
        try:
            # First assess if the topic is appropriate
            topic_assessment = self.phase_manager.assess_topic(user_input)
            
            if topic_assessment.relevance != TopicRelevance.ON_TOPIC:
                if topic_assessment.redirect_message:
                    self.display_manager.update_chat_display(
                        topic_assessment.redirect_message,
                        role="assistant"
                    )
                return
            
            # Get current differential diagnoses
            current_differential = []
            if self.differential_manager:
                current_differential = self.differential_manager.get_ranked_differential()
            
            # Get phase context for the message
            context = {
                "chat_history": [
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"].isoformat() if "timestamp" in msg else None
                    }
                    for msg in st.session_state.chat_messages
                ],
                "differential_diagnoses": [
                    {
                        "name": dx.name,
                        "order": idx + 1,
                        "notes": self.differential_manager.hypotheses[dx.name].notes
                    }
                    for idx, dx in enumerate(current_differential)
                ],
                "required_elements": {
                    "covered": [e.content for e in self.phase_manager.current_phase.required_elements if e.elicited],
                    "uncovered": [e.content for e in self.phase_manager.current_phase.required_elements if not e.elicited]
                },
                "completion_block_rationale": self.phase_manager.last_completion_block_rationale
            }
            
            # Log current required elements status
            self.logger.info("Required elements status:")
            for element in self.phase_manager.current_phase.required_elements:
                self.logger.info(f"- {element.content}: {'covered' if element.elicited else 'not covered'}")
            
            # Use the system prompt from session state
            system_prompt = st.session_state.get('current_phase_prompt')
            if not system_prompt:
                # If somehow missing, reconstruct it
                phase_context = self.phase_manager.get_phase_context()
                system_prompt = self.phase_manager.prompt_manager.construct_system_prompt(phase_context)
                st.session_state.current_phase_prompt = system_prompt
            
            # Get conversational response with full context
            response = self.llm_manager.get_conversational_response(
                system_prompt=system_prompt,
                user_message=json.dumps(context) + "\n\nUser message: " + user_input,
                message_history=st.session_state.chat_messages,
                temperature=0.7
            )
            
            if response:
                coverage_assessment = self.phase_manager.assess_coverage(user_input, response)
                self.display_manager.display_tutor_response(user_input, response)
                
                current_history = st.session_state.chat_messages + [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ]
                
                # Check and log phase completion status
                phase_status = getattr(st.session_state, 'phase_completion_status', {})
                current_phase = self.phase_manager.current_phase_type.value
                
                self.logger.info(f"Current phase: {current_phase}")
                self.logger.info(f"Phase completion status: {phase_status}")
                
                if not phase_status.get(current_phase, False):
                    self.logger.info("Checking phase completion...")
                    is_complete = self.phase_manager.check_phase_completion(current_history)
                    self.logger.info(f"Phase completion check result: {is_complete}")
                    
                    if is_complete:
                        self.logger.info("Phase complete! Displaying completion message...")
                        self._display_phase_completion_message()
                        return
                else:
                    self.logger.info("Phase already marked as complete")
                
                self._update_displays()
                st.rerun()
                
        except Exception as e:
            self.logger.error(f"Error handling user input: {str(e)}")
            st.error("An error occurred while processing your input. Please try again.")
            
    def _generate_next_response(self, coverage_assessment) -> str:
        """Generate the next appropriate response based on coverage assessment."""
        phase_context = self.phase_manager.get_phase_context()
        
        # First, check if we should present a teaching point
        if coverage_assessment.newly_covered_points:
            point = coverage_assessment.newly_covered_points[0]
            return self.prompt_manager.generate_teaching_prompt(
                point.content,
                self.phase_manager.current_phase_type
            )
        
        # If critical elements are missing, generate a probe for those
        if coverage_assessment.missing_critical_elements:
            return self.prompt_manager.construct_probe_question(
                coverage_assessment.missing_critical_elements[0],
                phase_context
            )
        
        # Otherwise, generate a general follow-up question
        return self.prompt_manager.construct_follow_up_question(phase_context)

    def _handle_phase_transition(self):
        """Handle transition to the next phase with enhanced summaries."""
        if not hasattr(self, 'phase_manager') or not self.phase_manager:
            self.logger.warning("Phase transition called before phase manager initialization")
            return
            
        if not hasattr(st.session_state, 'chat_messages'):
            self.logger.warning("Phase transition called before chat history initialization")
            return

        self.logger.info("Starting phase transition...")
        current_phase = self.phase_manager.current_phase_type
        
        # Generate comprehensive summary using chat history
        summaries = self.phase_manager.generate_phase_summary(
            st.session_state.chat_messages
        )
        
        # Store in session state for sidebar display
        if 'phase_summaries' not in st.session_state:
            st.session_state.phase_summaries = {}
        st.session_state.phase_summaries[current_phase] = summaries["phase_summary"]
        
        # Get next phase using phase sequence
        phase_sequence = [
            PhaseType.HISTORY,
            PhaseType.PHYSICAL,
            PhaseType.TESTING,
            PhaseType.MANAGEMENT,
            PhaseType.DISCUSSION
        ]
        
        try:
            current_idx = phase_sequence.index(current_phase)
            next_phase = phase_sequence[current_idx + 1] if current_idx + 1 < len(phase_sequence) else None
        except (ValueError, IndexError):
            self.logger.error(f"Invalid phase transition from {current_phase}")
            next_phase = None
        
        # Display the chat summary
        self.display_manager.update_chat_display(
            summaries["chat_summary"],
            role="assistant"
        )
        
        # Update the case information display with clinical summary
        self.display_manager.update_case_information(
            st.session_state.phase_summaries,
            current_phase
        )
        
        # If there's a next phase, add the transition prompt
        if next_phase:
            transition_prompt = f"\nClick the button below when you are ready to transition to the {next_phase.value.capitalize()} Phase"
            self.display_manager.update_chat_display(
                transition_prompt,
                role="assistant"
            )
            # Store next phase in session state for button handling
            st.session_state.pending_next_phase = next_phase
        
        self.logger.info(f"Phase transition complete. Next phase: {next_phase.value if next_phase else 'None'}")

    def _initialize_new_phase(self, new_phase: PhaseType):
        """Initialize everything needed for a new phase."""
        self.logger.info(f"Initializing new phase: {new_phase.value}")
        
        # Update phase in session state FIRST
        st.session_state.current_phase = new_phase
        
        # Only clear assessment cache if we're actually changing phases
        if not hasattr(st.session_state, 'last_phase') or st.session_state.last_phase != new_phase:
            st.session_state.assessment_cache = {}
            st.session_state.last_phase = new_phase
        
        # Reinitialize phase manager with new phase
        if hasattr(self, 'phase_manager'):
            self.phase_manager.current_phase_type = new_phase
            self.phase_manager._initialize_phase()
        
        # Clear chat messages except for case presentation
        if 'chat_messages' in st.session_state:
            initial_presentation = next(
                (msg for msg in st.session_state.chat_messages 
                if msg.get("is_presentation", False)), 
                None
            )
            st.session_state.chat_messages = [initial_presentation] if initial_presentation else []
        
        # Reset assessment cache for new phase
        st.session_state.assessment_cache = {}
        
        # Get the opening prompt from the newly initialized phase
        if hasattr(self.phase_manager.current_phase, 'config'):
            opening_prompt = self.phase_manager.current_phase.config.opening_prompt
            self.logger.info(f"Showing opening prompt for {new_phase.value}: {opening_prompt}")
            self.display_manager.update_chat_display(
                message=opening_prompt,
                role="assistant"
            )
    def _generate_next_prompt(self, coverage_assessment):
        """Generate the next appropriate prompt based on current context."""
        if coverage_assessment.newly_covered_points:
            point = coverage_assessment.newly_covered_points[0]
            self.display_manager.display_teaching_point(
                point.content,
                self.phase_manager.current_phase.config.teaching_guidance
            )
            
        phase_context = self.phase_manager.get_phase_context()
        next_prompt = self.prompt_manager.construct_system_prompt(phase_context)
        self.display_manager.update_chat_display(next_prompt)
    
    def _case_progress_bar(self, current_phase: str):
        phases = ["History", "Physical", "Testing", "Management", "Discussion"]
        phase_icons = {"Complete": "✓", "Current": "", "Future": ""}
        
        progress_bar = []
        for phase in phases:
            if phase == current_phase:
                progress_bar.append(f"<b>{phase} (current)</b>")  # Current phase
            elif phases.index(phase) < phases.index(current_phase):
                progress_bar.append(f"✓ {phase} (completed)")  # Completed phase
            else:
                progress_bar.append(f"{phase}")  # Future phase
        
        st.markdown(
            f"<div style='text-align: center; font-size: 18px; margin-top: 20px;'>"
            f"{'  →  '.join(progress_bar)}</div>",
            unsafe_allow_html=True,
        )
        
    def run(self):
        """Main application loop."""
        if hasattr(st.session_state, '_differential_updated'):
            del st.session_state._differential_updated
        
        self._case_progress_bar(st.session_state.current_phase.value.capitalize())
        st.text("")

        if st.session_state.case_loaded:
            # Ensure layout is set up
            if not self.display_manager.chat_col:
                self.display_manager._setup_layout()
                
            # Initialize initial display if needed
            if not st.session_state.chat_messages:
                self._update_initial_display()
                
            # Update all displays
            self._update_displays()
            
            # Handle chat interface
            with self.display_manager.chat_col:
                chat_container = st.container()
                with chat_container:
                    # Display chat messages
                    for msg in st.session_state.chat_messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    
                    # Add appropriate button based on state
                    phase_status = getattr(st.session_state, 'phase_completion_status', {})
                    current_phase = self.phase_manager.current_phase_type.value
                    
                    if phase_status.get(current_phase, False):
                        if not getattr(st.session_state, 'summary_generated', False):
                            if st.button(
                                "Generate Phase Summary",
                                key="generate_summary",
                                type="primary"
                            ):
                                self._generate_phase_summary()
                                st.rerun()
                        elif hasattr(st.session_state, 'pending_next_phase'):
                            next_phase = st.session_state.pending_next_phase
                            if st.button(
                                f"Proceed to {next_phase.value.capitalize()} Phase",
                                key=f"proceed_to_{next_phase.value}",
                                type="primary"
                            ):
                                self._initialize_new_phase(next_phase)
                                # Clear phase transition states
                                del st.session_state.pending_next_phase
                                del st.session_state.summary_generated
                                del st.session_state.phase_completion_status
                                st.rerun()
                
                # Handle user input at bottom of chat
                user_input = st.chat_input("Enter your response...")
                if user_input:
                    self._handle_user_input(user_input)
                    st.rerun()
        else:
            st.info("Please select a case to begin.")

if __name__ == "__main__":
    tutor = ClinicalCaseTutor()
    tutor.run()