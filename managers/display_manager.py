import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime
import logging
from streamlit.components.v1 import html
import json
import os

from models.case import CaseData, Diagnosis, DiagnosisCategory
from models.phase import PhaseType
from models.assessment import TopicAssessment, CoverageAssessment

class DisplayManager:
    def __init__(self, skip_page_config: bool = False):
        self.logger = logging.getLogger(__name__)
        
        # Initialize instance variables
        self.chat_col = None
        self.info_col = None
        self.case_container = None
        self.differential_container = None
        self.debug_container = None
        
        # Initialize session state
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "differential_diagnoses" not in st.session_state:
            st.session_state.differential_diagnoses = []
        if "debug_mode" not in st.session_state:
            st.session_state.debug_mode = False
        if "case_presented" not in st.session_state:
            st.session_state.case_presented = False
        
        if not skip_page_config:
            self._setup_layout()
    
    def _setup_layout(self):
        """Set up the main layout with chat and info panels side by side."""
        self.logger.info("Setting up layout")
        
        # Create main columns for chat and info
        cols = st.columns([2, 1])
        self.chat_col = cols[0]
        self.info_col = cols[1]
        
        # Set up info column with vertical sections
        with self.info_col:
            # Case Information section at the top
            st.markdown("### Case Information")
            self.case_container = st.container()
            
            # Differential Diagnosis section below case info
            st.markdown("### Differential Diagnosis")
            self.differential_container = st.container()
            
            # Debug section at the bottom
            st.checkbox("Debug Mode", key="debug_mode")
            if st.session_state.debug_mode:
                if st.button("Test Phase Transition"):
                    if hasattr(self, 'phase_transition_handler'):
                        self.phase_transition_handler()
                
                debug_cols = st.columns(3)
                with debug_cols[0]:
                    st.markdown("**Session State**")
                    st.write(st.session_state)
                with debug_cols[1]:
                    st.markdown("**Chat Messages**")
                    st.write(st.session_state.get('chat_messages', []))
                with debug_cols[2]:
                    st.markdown("**Differential**")
                    st.write(st.session_state.get('differential_diagnoses', []))


    def display_case_header(self, case_data: CaseData):
        """Display the case header information."""
        self.logger.info("Displaying case header")
        # Create header first
        st.title("Clinical Case Tutor")
        
        # Add phase progress if we have a current phase
        if hasattr(st.session_state, 'current_phase'):
            completed_phases = [
                phase for phase in PhaseType 
                if phase in st.session_state.get('phase_summaries', {})
            ]
            self.update_phase_progress(st.session_state.current_phase, completed_phases)
        
        # Create columns for main content
        col1, col2 = st.columns([2, 1])
        
        # Update the column references
        self.chat_col = col1
        self.info_col = col2
        
        # Display case info in the info column
        with self.info_col:
            self.case_container = st.container()
            with self.case_container:
                st.subheader(case_data.metadata.title)

    def update_phase_progress(self, current_phase: PhaseType, completed_phases: List[PhaseType]):
        """Display the phase sequence with completion status."""
        self.logger.info("Updating phase display")
        
        # Convert phase types to strings for JSON
        current = current_phase.value
        completed = [phase.value for phase in completed_phases]
        
        # Create component HTML
        component_html = f"""
        <div id="phase-progress-root"></div>
        <script>
            const root = document.getElementById('phase-progress-root');
            const progress = {{
                currentPhase: "{current}",
                completedPhases: {json.dumps(completed)}
            }};
            
            if (window.Streamlit) {{
                Streamlit.setComponentValue(JSON.stringify(progress));
            }}
        </script>
        """
        
        # Render the React component at the top of the page
        st.components.v1.html(
            component_html,
            height=100
        )

    def update_case_information(self, phase_summaries: Dict, current_phase: PhaseType):
        """Update the case information panel with clinical summaries."""
        with self.case_container:
            st.markdown("#### Case Summary")
            
            for phase_type in PhaseType:
                phase_summary = phase_summaries.get(phase_type)
                if phase_summary and phase_summary.get("findings_summary"):
                    tab_label = f"{phase_type.value.capitalize()} Findings"
                    with st.expander(tab_label, expanded=(phase_type == current_phase)):
                        clinical_summary = phase_summary["findings_summary"]
                        
                        # Display findings by category
                        for category, findings in clinical_summary.items():
                            if findings:  # Only show categories with content
                                st.markdown(f"**{category.title()}:**")
                                if isinstance(findings, list):
                                    for finding in findings:
                                        st.markdown(f"- {finding}")
                                elif isinstance(findings, dict):
                                    for key, value in findings.items():
                                        st.markdown(f"- **{key}:** {value}")
                                else:
                                    st.markdown(f"- {findings}")
                        
                        # If it's the current phase and debug mode is on, show learner assessment
                        # in a separate section below the findings
                        if phase_type == current_phase and st.session_state.get("debug_mode", False):
                            st.markdown("---")
                            st.markdown("**🔍 Learner Assessment (Debug)**")
                            st.json(phase_summary.get("learner_assessment", {}))

    def update_differential_panel(self, differential_manager):
        """Update the differential diagnosis panel."""
        if not self.differential_container:
            self._setup_layout()
                    
        with self.differential_container:
            try:
                if differential_manager:
                    current_differential = differential_manager.get_ranked_differential()
                    
                    # CSS styling
                    st.markdown("""
                        <style>
                        div[data-testid="stExpander"] {
                            background: transparent !important;
                            border: none !important;
                            box-shadow: none !important;
                            margin: 0px 0 !important;
                            padding-left: 12px !important;
                        }
                        
                        div[data-testid="stExpander"] > div {
                            background-color: #1E1E1E !important;
                            border: 1px solid #444 !important;
                            border-radius: 8px !important;
                            box-shadow: none !important;
                        }
                        
                        div[data-testid="stExpander"] div[data-testid="stExpanderHeader"] {
                            background: transparent !important;
                            border: none !important;
                        }
                        
                        .arrow-button {
                            background-color: transparent !important;
                            border: 1px solid #444 !important;
                            border-radius: 4px !important;
                            color: #fff !important;
                            padding: 2px 8px !important;
                            font-size: 14px !important;
                            height: 24px !important;
                            width: 24px !important;
                            min-height: 24px !important;
                            line-height: 1 !important;
                            margin: 2px !important;
                            cursor: pointer !important;
                        }
                        
                        .arrow-button:hover {
                            background-color: rgba(255, 255, 255, 0.1) !important;
                        }
                        
                        .add-button {
                            background-color: rgb(28, 30, 33) !important;
                            border: 1px solid #444 !important;
                            border-radius: 8px !important;
                            color: #fff !important;
                            padding: 0.5rem 1rem !important;
                            width: 100% !important;
                            height: 38px !important;
                            margin-top: 8px !important;
                            margin-bottom: 16px !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                    
                    # Add new diagnosis input - using container instead of columns for sidebar compatibility
                    input_container = st.container()
                    
                    # Define the on_input_change function inside update_differential_panel
                    def on_input_change():
                        input_key = f"new_dx_input_{id(self.differential_container)}"
                        if st.session_state[input_key].strip():  # Only process non-empty input
                            differential_manager.add_user_diagnosis(st.session_state[input_key])
                            st.session_state[input_key] = ""  # Clear the input
                            #st.rerun()
                    
                    # Create the input field using the full width with on_change handler
                    new_dx = input_container.text_input(
                        "Add new diagnosis",
                        key=f"new_dx_input_{id(self.differential_container)}",
                        label_visibility="collapsed",
                        on_change=on_input_change  # Handle Enter key
                    )
                    
                    # Add button below the input
                    if input_container.button(
                        "Add",
                        key=f"add_dx_button_{id(self.differential_container)}",
                        kwargs={"className": "add-button"}
                    ):
                        if new_dx.strip():  # Only process non-empty input
                            differential_manager.add_user_diagnosis(new_dx)
                            # Clear the input
                            st.session_state[f"new_dx_input_{id(self.differential_container)}"] = ""
                            st.rerun()
                    
                    # Display existing diagnoses
                    if len(current_differential) > 0:
                        for index, dx in enumerate(current_differential):
                            rank = index + 1
                            
                            # Create a container for each diagnosis
                            dx_container = st.container()
                            
                            # Create a row with arrows, rank, title, and delete button
                            row_cols = dx_container.columns([0.5, 0.5, 4, 0.5])
                            
                            # Up arrow
                            with row_cols[0]:
                                if index > 0:
                                    if st.button("↑", key=f"up_{dx.name}_{rank}", help="Move up", 
                                            kwargs={"className": "arrow-button"}):
                                        prev_dx = current_differential[index - 1]
                                        differential_manager.swap_diagnoses(dx.name, prev_dx.name)
                                        st.rerun()
                            
                            # Down arrow
                            with row_cols[1]:
                                if index < len(current_differential) - 1:
                                    if st.button("↓", key=f"down_{dx.name}_{rank}", help="Move down", 
                                            kwargs={"className": "arrow-button"}):
                                        next_dx = current_differential[index + 1]
                                        differential_manager.swap_diagnoses(dx.name, next_dx.name)
                                        st.rerun()
                            
                            # Diagnosis title and expander
                            with row_cols[2]:
                                with st.expander(f"#{rank}: {dx.name}", expanded=False):
                                    notes = differential_manager.hypotheses[dx.name].notes
                                    new_notes = st.text_area(
                                        "Notes", 
                                        value=notes, 
                                        key=f"notes_{dx.name}_{rank}"
                                    )
                                    if new_notes != notes:
                                        differential_manager.update_diagnosis_notes(dx.name, new_notes)
                                        st.rerun()
                            
                            # Delete button
                            with row_cols[3]:
                                if st.button("🗑️", key=f"delete_{dx.name}_{rank}", help="Remove diagnosis"):
                                    differential_manager.remove_diagnosis(dx.name)
                                    st.rerun()
                    else:
                        st.info("No diagnoses added yet. Add your first diagnosis above.")

            except Exception as e:
                self.logger.error(f"Error updating differential panel: {str(e)}")
                st.error(f"Error updating differential diagnoses: {str(e)}")

    def display_phase_transition(self, from_phase: PhaseType, to_phase: PhaseType, summary: Dict):
        """Display phase transition information."""
        self.logger.info(f"Displaying phase transition from {from_phase} to {to_phase}")
        
        transition_message = (
            f"**Completing {from_phase.value.capitalize()} Phase**\n\n"
            f"Key Points Covered:\n"
            + "\n".join(f"- {point}" for point in summary.get("covered_points", []))
            + f"\n\n**Beginning {to_phase.value.capitalize()} Phase**"
        )
        
        self.update_chat_display(transition_message, "assistant")

    def get_user_input(self) -> Optional[str]:
        """Get user input from the chat interface."""
        self.logger.info("Getting user input")
        return st.chat_input("Enter your response...")

    def display_chat_messages(self):
        """Display all chat messages."""
        if not hasattr(st.session_state, 'chat_messages'):
            st.session_state.chat_messages = []
            
        with self.chat_col:
            # Create a container for messages if it doesn't exist
            if not hasattr(self, 'message_container'):
                self.message_container = st.container()
                
            with self.message_container:
                # Clear existing messages
                self.message_container.empty()
                
                # Display messages
                for msg in st.session_state.chat_messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

    def update_chat_display(self, message: str, role: str = "assistant", clear_input: bool = True):
        """Update the chat display with a new message."""
        self.logger.info(f"Updating chat display with message from {role}")
        
        if role != "system":
            message_dict = {
                "role": role,
                "content": message,
                "timestamp": datetime.now()
            }
            
            # Check for duplicates
            if not any(
                msg["content"] == message and msg["role"] == role 
                for msg in st.session_state.chat_messages
            ):
                st.session_state.chat_messages.append(message_dict)
                # Remove the display update - let the main run loop handle it
    
    def display_tutor_response(self, user_input: str, response: str):
        """Add user input and tutor response to chat history."""
        if not hasattr(st.session_state, 'chat_messages'):
            st.session_state.chat_messages = []
            
        # Add user message if not already present
        user_dict = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        }
        
        # Add assistant response if not already present
        assistant_dict = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        }
        
        # Update session state
        st.session_state.chat_messages.extend([user_dict, assistant_dict])
        # Remove the display_chat_messages call - let the main run loop handle it

    def display_chat_messages(self):
        """
        This method is now deprecated and should not be called directly.
        Chat display is handled by the main run loop.
        """
        self.logger.warning("display_chat_messages called directly - this method is deprecated")
        pass

    def display_teaching_point(self, point: str, context: Optional[str] = None):
        """Add a teaching point to chat history."""
        self.logger.info("Displaying teaching point")
        
        teaching_message = f"**Key Concept:**\n{point}"
        if context:
            teaching_message += f"\n\n**Context:**\n{context}"
            
        self.update_chat_display(teaching_message, "assistant")
        # Let the main run loop handle the display

    def handle_redirection(self, assessment: TopicAssessment):
        """Display redirection message when user goes off-topic."""
        self.logger.info("Handling redirection")
        if assessment.redirect_message:
            with self.chat_col:
                with st.chat_message("assistant"):
                    st.markdown(assessment.redirect_message)

    def toggle_debug(self):
        """Toggle debug mode for differential panel."""
        if st.session_state.get('debug_mode'):
            with self.differential_container:
                st.write("Debug Info:")
                st.write("Session State Diagnoses:", st.session_state.get('saved_diagnoses', []))
                if hasattr(self, 'differential_manager'):
                    st.write("Current Hypotheses:", list(self.differential_manager.hypotheses.keys()))

    def display_debug_information(
        self,
        phase_data: Dict = None,
        coverage_data: Dict = None,
        assessment_data: Dict = None,
        case_data: Optional[CaseData] = None,
        case_state: Optional[Dict] = None 
    ):
        """Display debug information when debug mode is enabled."""
        if not st.session_state.get("debug_mode", False):
            return
            
        with self.debug_container:
            if phase_data:
                with st.expander("Phase Data", expanded=True):
                    st.json(phase_data)
            
            if coverage_data:
                with st.expander("Coverage Data", expanded=True):
                    st.json(coverage_data)
            
            if assessment_data:
                with st.expander("Latest Assessment", expanded=True):
                    st.json(assessment_data)
            
            if case_data:
                try:
                    case_data_dict = {
                        "metadata": case_data.metadata.__dict__,
                        "phases": {phase_type.value: phase.__dict__ for phase_type, phase in case_data.phases.items()},
                        "differential_diagnosis": [dx.__dict__ for dx in case_data.differential_diagnosis],
                        "final_diagnosis": case_data.final_diagnosis.__dict__
                    }
                    if case_state:
                        case_data_dict["case_state"] = case_state
                        
                    with st.expander("Case Data", expanded=True):
                        st.json(case_data_dict)
                except Exception as e:
                    self.logger.error(f"Error serializing case data: {str(e)}")
                    st.error(f"Error displaying case data: {str(e)}")