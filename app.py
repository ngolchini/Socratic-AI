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
from utils.case_importer import import_cases_from_csv

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

        # Retrieve the OpenAI API key from Streamlit secrets
        api_key = st.secrets["api"]["OPENAI_API_KEY"]
        if not api_key:
            st.error("OpenAI API key not found in secrets. Please set it in the Streamlit secrets configuration.")
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
        
        # Only set up case if not showing search page
        if not st.session_state.get('show_search_page', True):
            self._setup_case()
            
        # Initialize additional session state variables for search
        if "last_search" not in st.session_state:
            st.session_state.last_search = ""
        if "applied_filters" not in st.session_state:
            st.session_state.applied_filters = {}
    
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
            st.session_state.differential_manager = None
            st.session_state._session_id = 0
            st.session_state.show_import_dialog = False
            st.session_state.show_search_page = True

    def _initialize_managers(self):
        """Initialize all managers with current session state."""
        if not hasattr(st.session_state, 'case_data') or not st.session_state.case_data:
            return
            
        self.phase_manager = PhaseManager(
            case_data=st.session_state.case_data,
            llm_manager=self.llm_manager,
            prompt_manager=self.prompt_manager
        )
        
        # Only create a new DifferentialManager if one doesn't exist
        if st.session_state.differential_manager is None:
            self.differential_manager = DifferentialManager(self.llm_manager)
            st.session_state.differential_manager = self.differential_manager
        else:
            self.differential_manager = st.session_state.differential_manager

    def _get_available_cases(self) -> list:
        """Get list of available clinical cases."""
        cases_dir = Path("cases")
        return [f.stem for f in cases_dir.glob("*.json")]
    
    def _load_new_case(self, case_id: str):
        """Load a new clinical case and initialize its managers."""
        try:
            self.logger.info(f"Loading case: {case_id}")
            
            # Only proceed if this is actually a new case
            if st.session_state.current_case_id == case_id:
                return
                
            # Load case data first
            case_data = self.case_manager.load_case(case_id)
            self.logger.info(f"Case loaded successfully: {case_id}")
            
            # Reset all state in a single block
            st.session_state.chat_messages = []
            st.session_state.assessment_cache = {}
            st.session_state.phase_summaries = {}
            st.session_state.differential_manager = None
            st.session_state.case_data = case_data
            st.session_state.current_case_id = case_id
            st.session_state.current_phase = PhaseType.HISTORY
            st.session_state.case_loaded = True  # Make sure this is set to True
            st.session_state.case_presented = False
            st.session_state.show_search_page = False  # Explicitly set this to False
            
            # Clear any existing phase-related flags
            if 'phase_completion_status' in st.session_state:
                del st.session_state.phase_completion_status
            if 'summary_generated' in st.session_state:
                del st.session_state.summary_generated
            if 'pending_next_phase' in st.session_state:
                del st.session_state.pending_next_phase
                
            # Reinitialize managers with new case data
            self._initialize_managers()
                
            # Force a complete rerun with new session ID to ensure fresh state
            st.session_state['_session_id'] = st.session_state.get('_session_id', 0) + 1
            
            # Debug log to confirm the case is loaded
            self.logger.info(f"Case {case_id} loaded, rerunning app...")
            
            # Use this for debugging
            st.session_state.debug_message = f"Loaded case: {case_id}"
            
            st.rerun()
                
        except Exception as e:
            self.logger.error(f"Error loading case: {str(e)}")
            st.error(f"Error loading case: {str(e)}")

    def _setup_case(self):
        """Set up or continue the current clinical case."""
        # If show_search_page is True, don't do anything here
        if st.session_state.get('show_search_page', True):
            return
            
        available_cases = self._get_available_cases()
        
        # with st.sidebar:
        #     # Search input field
        #     search_query = st.text_input("🔍 Search Cases", key="case_search")
            
        #     if search_query:
        #         # Perform search
        #         search_results = self.case_manager.search_cases(search_query)
        #         if search_results:
        #             # Create a descriptive list of search results
        #             case_options = []
        #             case_id_map = {}
                    
        #             for case in search_results:
        #                 case_id = case["id"]
        #                 display_text = f"{case_id}: {case['title']}"
        #                 case_options.append(display_text)
        #                 case_id_map[display_text] = case_id
                    
        #             # Use selectbox for search results
        #             selected_option = st.selectbox(
        #                 "Search Results", 
        #                 case_options,
        #                 key=f"search_results_{st.session_state.get('_session_id', 0)}"
        #             )
                    
        #             selected_case = case_id_map[selected_option]
                    
        #             # Only load if the selection changed
        #             if selected_case != st.session_state.current_case_id:
        #                 self._load_new_case(selected_case)
        #         else:
        #             st.info("No matching cases found.")
        #             # Show all cases as fallback
        #             self._display_regular_case_selector(available_cases)
        #     else:
        #         # Regular case selector when not searching
        #         self._display_regular_case_selector(available_cases)
        
        # def _display_regular_case_selector(self, available_cases):
        #     """Display the regular case selector dropdown."""
        #     # Use a timestamp-based key to ensure fresh state
        #     selectbox_key = f"case_selector_{st.session_state.get('_session_id', 0)}"
            
        #     current_index = 0
        #     if st.session_state.current_case_id in available_cases:
        #         current_index = available_cases.index(st.session_state.current_case_id)
                
        #     selected_case = st.selectbox(
        #         "Select Clinical Case",
        #         options=available_cases,
        #         index=current_index,
        #         key=selectbox_key
        #     )
            
        #     # Only trigger case load if selection actually changed
        #     if selected_case != st.session_state.current_case_id:
        #         self._load_new_case(selected_case)

    def _display_regular_case_selector(self, available_cases):
        """Display the regular case selector dropdown."""
        # Use a timestamp-based key to ensure fresh state
        selectbox_key = f"case_selector_{st.session_state.get('_session_id', 0)}"
        
        current_index = 0
        if st.session_state.current_case_id in available_cases:
            current_index = available_cases.index(st.session_state.current_case_id)
            
        selected_case = st.selectbox(
            "Select Clinical Case",
            options=available_cases,
            index=current_index,
            key=selectbox_key
        )
        
        # Only trigger case load if selection actually changed
        if selected_case != st.session_state.current_case_id:
            self._load_new_case(selected_case)

    def _show_import_dialog(self):
        """Show dialog to import cases from CSV."""
        st.sidebar.subheader("Import Cases")
        
        csv_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        
        case_limit = st.sidebar.number_input("Number of cases to import", min_value=1, max_value=100, value=50)
        
        if st.sidebar.button("Start Import"):
            if csv_file:
                # Save the uploaded file temporarily
                temp_csv_path = Path("temp_cases.csv")
                with open(temp_csv_path, "wb") as f:
                    f.write(csv_file.getbuffer())
                
                try:
                    # Import the cases
                    imported_cases = import_cases_from_csv(str(temp_csv_path), "cases", limit=case_limit)
                    st.sidebar.success(f"Successfully imported {len(imported_cases)} cases!")
                    
                    # Clean up the temp file
                    temp_csv_path.unlink()
                    
                    # Reset session to show new cases
                    st.session_state._session_id = st.session_state.get('_session_id', 0) + 1
                    
                    # Close the dialog
                    st.session_state.show_import_dialog = False
                    
                    # Force a rerun to refresh the UI
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"Error importing cases: {str(e)}")
            else:
                st.sidebar.error("Please upload a CSV file.")
        
        if st.sidebar.button("Cancel"):
            st.session_state.show_import_dialog = False
            st.rerun()
            
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
    def _update_initial_display(self):
        """Initialize all display components with current case state."""
        if st.session_state.case_data:
            self.logger.info("Updating initial display with case data")
            self.display_manager.display_case_header(st.session_state.case_data)
            self._update_displays()
            self._display_initial_prompt()
        else:
            self.logger.error("No case data found in session state")
    
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
        self.logger.info(f"Handling user input: {user_input}")
        
        if not st.session_state.case_loaded or not self.phase_manager:
            st.error("Please select a case before continuing.")
            return
        
        # Debug shortcut for phase transition
        if user_input.lower() in ["move forward", "next phase", "skip"]:
            self.logger.info("Debug command detected - forcing phase transition")
            self.display_manager.update_chat_display(
                "⚡ Debug command detected - moving to phase transition...",
                role="assistant"
            )
            self._handle_phase_transition()
            return
                
        try:
            # First assess if the topic is appropriate
            topic_assessment = self.phase_manager.assess_topic(user_input)
            self.logger.info(f"Topic assessment result: {topic_assessment}")
            
            if topic_assessment.relevance != TopicRelevance.ON_TOPIC:
                if topic_assessment.redirect_message:
                    self.display_manager.update_chat_display(
                        topic_assessment.redirect_message,
                        role="assistant"
                    )
                return
            
            # Add user message to chat history first
            self.display_manager.update_chat_display(
                message=user_input,
                role="user"
            )
            
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
                        "notes": self.differential_manager.hypotheses[dx.name].notes if dx.name in self.differential_manager.hypotheses else ""
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
            self.logger.info("Getting LLM response...")
            response = self.llm_manager.get_conversational_response(
                system_prompt=system_prompt,
                user_message=json.dumps(context) + "\n\nUser message: " + user_input,
                message_history=st.session_state.chat_messages,
                temperature=0.7
            )
            
            self.logger.info(f"LLM response received: {response is not None}")
            
            if response:
                # Add bot response to chat history
                self.display_manager.update_chat_display(
                    message=response,
                    role="assistant"
                )
                
                # Assess coverage and check completion
                coverage_assessment = self.phase_manager.assess_coverage(user_input, response)
                
                # Update phase completion status if needed
                phase_status = getattr(st.session_state, 'phase_completion_status', {})
                current_phase = self.phase_manager.current_phase_type.value
                
                if not phase_status.get(current_phase, False):
                    is_complete = self.phase_manager.check_phase_completion(st.session_state.chat_messages)
                    if is_complete:
                        self.logger.info("Phase complete! Displaying completion message...")
                        self._display_phase_completion_message()
                        return
            else:
                self.logger.error("No response received from LLM")
                st.error("I apologize, but I wasn't able to generate a response. Please try again.")
                return
                
            self._update_displays()
        
        except Exception as e:
            self.logger.error(f"Error handling user input: {str(e)}", exc_info=True)
            st.error("An error occurred while processing your input. Please try again.")
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
        """Handle transition to the next phase with enhanced summaries and differential check."""
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
        
       # Check differential diagnosis if we're transitioning from certain phases
        if current_phase in [PhaseType.HISTORY, PhaseType.PHYSICAL, PhaseType.TESTING]:
            try:
                # Get the current phase object
                phase = self.phase_manager.current_phase
                ideal_differential = phase.current_ideal_differential_diagnosis
                
                if ideal_differential and self.differential_manager:
                    matches_sufficiently, feedback = self.differential_manager.compare_differentials(ideal_differential)
                    
                    # Display feedback about differential
                    self.display_manager.update_chat_display(
                        feedback,
                        role="assistant"
                    )
                    
                    # If differential doesn't match well enough, pause transition
                    if not matches_sufficiently:
                        self.display_manager.update_chat_display(
                            "Please review and update your differential diagnosis based on the feedback above before proceeding.",
                            role="assistant"
                        )
                        return
            except Exception as e:
                self.logger.error(f"Error comparing differentials: {str(e)}")
                # Continue with phase transition even if differential comparison fails
        
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
        if hasattr(st.session_state, '_differential_updated'):
            del st.session_state._differential_updated
        
        # Show search page if no case is selected yet or if explicitly requested
        if (not st.session_state.get('case_loaded', False) or 
            not st.session_state.get('case_data') or 
            st.session_state.get('show_search_page', True)):
            
            self._show_search_page()
            return
        # Add Home button to sidebar

        with st.sidebar:
            # Clear out the previous sidebar content
            st.empty()
            
            # Add home button
            if st.button("🏠 Home", help="Return to case selection"):
                # Reset case state
                st.session_state.show_search_page = True
                # Don't clear the case data yet - just show the search page
                st.rerun()
                
            # You can add other sidebar content here that's specific to the case view
            # For example, you might want to show case metadata
            if st.session_state.case_data and hasattr(st.session_state.case_data, 'metadata'):
                st.subheader("Case Info")
                metadata = st.session_state.case_data.metadata
                st.write(f"**ID:** {metadata.id}")
                if hasattr(metadata, 'difficulty'):
                    st.write(f"**Difficulty:** {metadata.difficulty}")
                if hasattr(metadata, 'specialties') and metadata.specialties:
                    st.write(f"**Specialty:** {', '.join(metadata.specialties)}")
    
        # If a case is loaded, continue with the normal flow
        self._case_progress_bar(st.session_state.current_phase.value.capitalize())
        st.text("")
        
        # Ensure managers are initialized
        self._initialize_managers()
        
        # Setup layout and continue with normal flow
        if not self.display_manager.chat_col:
            self.display_manager._setup_layout()
            
        if not st.session_state.chat_messages:
            self._update_initial_display()
            
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

    def _show_search_page(self):
        """Show the case search page as the initial screen."""
        st.title("Clinical Case Tutor")
        st.subheader("Search Medical Cases")
        
        # Large search input at the top
        search_query = st.text_input(
            "Enter search terms (symptoms, conditions, specialties, etc.)",
            key="main_search",
            value=st.session_state.get("last_search", ""),
            placeholder="Example: 'dyspnea chest pain fever' or 'cardiology'",
        )
        
        available_cases = self._get_available_cases()
        
        if search_query:
            # Store last search query
            st.session_state.last_search = search_query
            
            # Perform search
            search_results = self.case_manager.search_cases(search_query, top_k=15)
            
            if search_results:
                st.subheader(f"Found {len(search_results)} matching cases:")
                
                # Create columns for better display
                col1, col2 = st.columns([7, 3])
                
                with col1:
                    # Display cases as selectable cards
                    for i, case in enumerate(search_results):
                        case_id = case["id"]
                        title = case["title"]
                        specialty_text = ", ".join(case.get("specialties", []))
                        difficulty = case.get("difficulty", "Intermediate")
                        
                        # Create a card-like display for each case
                        with st.container(border=True):
                            st.markdown(f"**{title}**")
                            st.markdown(f"Specialties: {specialty_text}")
                            st.markdown(f"Difficulty: {difficulty}")
                            
                            # Button to select this case
                            if st.button(f"Start Case", key=f"select_{case_id}"):
                                self._load_new_case(case_id)
                                st.rerun()
                
                with col2:
                    st.subheader("Quick Filters")
                    
                    # Extract all specialties from search results
                    all_specialties = []
                    for case in search_results:
                        all_specialties.extend(case.get("specialties", []))
                    
                    unique_specialties = sorted(list(set(all_specialties)))
                    
                    # Filter by specialty
                    selected_specialty = st.selectbox(
                        "Filter by Specialty",
                        options=["All Specialties"] + unique_specialties,
                        key="specialty_filter"
                    )
                    
                    # Filter by difficulty
                    difficulty_options = ["All Difficulties", "Basic", "Intermediate", "Advanced"]
                    selected_difficulty = st.selectbox(
                        "Filter by Difficulty",
                        options=difficulty_options,
                        key="difficulty_filter"
                    )
                    
                    # Apply filters button
                    if st.button("Apply Filters"):
                        # Implementation would filter the displayed results
                        st.session_state.applied_filters = {
                            "specialty": selected_specialty if selected_specialty != "All Specialties" else None,
                            "difficulty": selected_difficulty if selected_difficulty != "All Difficulties" else None
                        }
                        st.rerun()
                    
                    # Reset filters
                    if st.button("Reset Filters"):
                        st.session_state.applied_filters = {}
                        st.rerun()
            else:
                st.info("No matching cases found. Try different search terms.")
                
                # Show a few random cases as suggestions
                st.subheader("You might be interested in:")
                import random
                sample_size = min(5, len(available_cases))
                random_cases = random.sample(available_cases, sample_size)
                
                for case_id in random_cases:
                    try:
                        case_path = Path("cases") / f"{case_id}.json"
                        with open(case_path, 'r') as f:
                            case_data = json.load(f)
                        metadata = case_data.get('metadata', {})
                        
                        with st.container(border=True):
                            st.markdown(f"**{metadata.get('title', case_id)}**")
                            if 'specialties' in metadata:
                                st.markdown(f"Specialties: {', '.join(metadata['specialties'])}")
                            
                            if st.button(f"Start Case", key=f"random_{case_id}"):
                                self._load_new_case(case_id)
                                st.rerun()
                    except Exception as e:
                        self.logger.error(f"Error loading case {case_id}: {str(e)}")
        else:
            # When no search is performed, show featured or categorized cases
            st.subheader("Featured Cases")
            
            # Create a tabbed interface for categories
            tabs = st.tabs(["Recent", "Popular", "All Cases"])
            
            with tabs[0]:  # Recent tab
                st.write("Recently added cases:")
                # Show most recent cases based on file creation date
                recent_cases = sorted(
                    available_cases, 
                    key=lambda x: os.path.getctime(Path("cases") / f"{x}.json"),
                    reverse=True
                )[:10]
                
                for case_id in recent_cases[:5]:
                    try:
                        case_path = Path("cases") / f"{case_id}.json"
                        with open(case_path, 'r') as f:
                            case_data = json.load(f)
                        metadata = case_data.get('metadata', {})
                        
                        with st.container(border=True):
                            st.markdown(f"**{metadata.get('title', case_id)}**")
                            if 'specialties' in metadata:
                                st.markdown(f"Specialties: {', '.join(metadata['specialties'])}")
                            
                            if st.button(f"Start Case", key=f"recent_{case_id}"):
                                self._load_new_case(case_id)
                                st.rerun()
                    except Exception as e:
                        self.logger.error(f"Error loading case {case_id}: {str(e)}")
            
            with tabs[1]:  # Popular tab
                st.write("Popular cases:")
                # In a real app, this would be based on usage data
                # For now, just show some cases as examples
                import random
                sample_cases = random.sample(available_cases, min(5, len(available_cases)))
                
                for case_id in sample_cases:
                    # Similar display logic as above
                    try:
                        case_path = Path("cases") / f"{case_id}.json"
                        with open(case_path, 'r') as f:
                            case_data = json.load(f)
                        metadata = case_data.get('metadata', {})
                        
                        with st.container(border=True):
                            st.markdown(f"**{metadata.get('title', case_id)}**")
                            if 'specialties' in metadata:
                                st.markdown(f"Specialties: {', '.join(metadata['specialties'])}")
                            
                            if st.button(f"Start Case", key=f"popular_{case_id}"):
                                self._load_new_case(case_id)
                                st.rerun()
                    except Exception as e:
                        self.logger.error(f"Error loading case {case_id}: {str(e)}")
            
            with tabs[2]:  # All Cases tab
                st.write("All available cases:")
                
                # Group cases by specialty for better organization
                cases_by_specialty = {}
                
                for case_id in available_cases:
                    try:
                        case_path = Path("cases") / f"{case_id}.json"
                        with open(case_path, 'r') as f:
                            case_data = json.load(f)
                        metadata = case_data.get('metadata', {})
                        specialties = metadata.get('specialties', ['Uncategorized'])
                        
                        for specialty in specialties:
                            if specialty not in cases_by_specialty:
                                cases_by_specialty[specialty] = []
                            cases_by_specialty[specialty].append((case_id, metadata.get('title', case_id)))
                    except Exception as e:
                        self.logger.error(f"Error loading case {case_id}: {str(e)}")
                
                # Display cases by specialty
                # Add a unique key to avoid duplicates
                for specialty_idx, (specialty, cases) in enumerate(sorted(cases_by_specialty.items())):
                    with st.expander(f"{specialty} ({len(cases)} cases)"):
                        for case_idx, (case_id, title) in enumerate(cases):
                            cols = st.columns([4, 1])
                            with cols[0]:
                                st.write(title)
                            with cols[1]:
                                # Use a more unique key that includes the specialty index
                                if st.button("Select", key=f"all_{specialty_idx}_{case_id}_{case_idx}"):
                                    self._load_new_case(case_id)
                                    st.rerun()

if __name__ == "__main__":
    tutor = ClinicalCaseTutor()
    tutor.run()
