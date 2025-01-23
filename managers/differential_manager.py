from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging
import json
import streamlit as st

from models.case import Diagnosis, DiagnosisCategory
from models.phase import PhaseType, ClinicalElement
from managers.llm_manager import LLMManager

@dataclass
class DiagnosticEvidence:
    """Tracks evidence for a specific diagnosis."""
    timestamp: datetime
    phase: PhaseType
    content: str
    strength: float  # 0.0 to 1.0
    supports: bool = True

@dataclass
class DiagnosticHypothesis:
    """Represents a diagnostic hypothesis with its evidence trail and user notes."""
    diagnosis: Diagnosis
    proposed_at: datetime
    evidence: List[DiagnosticEvidence] = field(default_factory=list)
    notes: str = ""
    order_index: int = 0  # For tracking user-defined ordering

class DifferentialManager:
    """Manages user-created differential diagnoses throughout the case."""
    
    def __init__(self, llm_manager: LLMManager):
        self.hypotheses: Dict[str, DiagnosticHypothesis] = {}
        self.llm_manager = llm_manager
        self.logger = logging.getLogger(__name__)

    def add_user_diagnosis(
        self,
        name: str,
        category: DiagnosisCategory = DiagnosisCategory.POSSIBLE,
        notes: str = "",
        order_index: Optional[int] = None
    ) -> DiagnosticHypothesis:
        """Add a new diagnosis from user input."""
        self.logger.info(f"Adding user diagnosis: {name}")
        
        # Check if diagnosis already exists
        if name in self.hypotheses:
            self.logger.info(f"Diagnosis {name} already exists")
            return self.hypotheses[name]
        
        # Create new diagnosis
        diagnosis = Diagnosis(
            name=name,
            category=category,
            key_features=[],
            supporting_evidence=[],
            refuting_evidence=[]
        )
        
        # Calculate new order_index if not provided
        if order_index is None:
            order_index = len(self.hypotheses)
        
        # Add to hypotheses
        hypothesis = DiagnosticHypothesis(
            diagnosis=diagnosis,
            proposed_at=datetime.now(),
            notes=notes,
            order_index=order_index
        )
        
        self.hypotheses[name] = hypothesis
        self.logger.info(f"Added new diagnosis. Total diagnoses: {len(self.hypotheses)}")
        
        # Save state
        if not hasattr(st.session_state, 'saved_diagnoses'):
            st.session_state.saved_diagnoses = []
        
        st.session_state.saved_diagnoses = self.save_state()
        self.logger.info("Saved diagnoses to session state")
        
        return hypothesis

    def update_diagnosis_notes(self, diagnosis_name: str, notes: str):
        """Update the notes for a specific diagnosis."""
        if diagnosis_name in self.hypotheses:
            self.hypotheses[diagnosis_name].notes = notes
            self.logger.info(f"Updated notes for {diagnosis_name}")

    def compare_differentials(
        self,
        ideal_differential: List[Dict]
    ) -> Tuple[bool, str]:
        """
        Compare user's differential with the ideal differential for the current phase
        
        Args:
            ideal_differential: List of ideal diagnoses from LLM
        
        Returns:
            Tuple of (matches_sufficiently: bool, feedback_message: str)
        """
        system_prompt = """You are a medical educator assessing a learner's differential diagnosis against an expert-generated differential.
        The goal is to take the differential they have created and use it to provide constructive feedback WITHOUT telling them the answer, as 
        they will be moving to the next step of the case. 
        Consider:
        1. What percentage of the key diagnoses are present (recall)? 
        2. Is the ranking/prioritization reasonable?
        3. Of the missed diagnosis, what are clues that you can give to help them think about them for next steps?
        4. Are there any inappropriate inclusions?
        
        Return a JSON object with:
        {
            "1. recall rate: float,
            "2. ranking order": [string],
            "3. Clues for missed diagnoses": [string],
            "4. Inappropriate inclusions and why": [string]
        }
        """
        
        # Get current user differential
        user_differential = self.get_ranked_differential()
        
        # Format the differentials for comparison
        comparison_data = {
            "user_differential": [
                {
                    "name": dx.name,
                    "category": dx.category.value,
                    "key_features": dx.key_features,
                    "order": idx + 1
                }
                for idx, dx in enumerate(user_differential)
            ],
            "ideal_differential": ideal_differential
        }
        
        response = self.llm_manager.get_json_response(
            system_prompt,
            json.dumps(comparison_data)
        )
        
        # Create a structured summary of both differentials
        comparison_summary = """
    ## 🎯 Differential Diagnosis Comparison

    ### Your Current Differential:
    {}

    ### Expert Differential for this Stage:
    {}

    ### Analysis:
    {}

    {}

    {}

    ### Overall Assessment:
    ✨ Sufficient Match: {}
        """.format(
            "\n".join(f"- {idx+1}. {dx.name}" for idx, dx in enumerate(user_differential)),
            "\n".join(f"- {dx['name']} ({dx['likelihood']})" for dx in ideal_differential),
            response.get('feedback', 'No specific feedback available.'),
            f"🚩 **Missing Key Diagnoses**: {', '.join(response.get('missing_key_diagnoses', []))}" if response.get('missing_key_diagnoses') else "",
            f"⚠️ **Ranking Feedback**: {response.get('ranking_feedback', '')}" if response.get('ranking_feedback') else "",
            "✅ Yes" if response.get("sufficient_match", False) else "❌ No"
        )
        
        self.logger.info(f"Differential comparison completed. Match sufficient: {response.get('sufficient_match', False)}")
        
        return (
            response.get("sufficient_match", False),
            comparison_summary
        )


    def update_diagnosis_order(self, diagnosis_name: str, new_index: int):
        """Update the order index for a specific diagnosis."""
        if diagnosis_name not in self.hypotheses:
            return
                
        ordered_diagnoses = sorted(
            self.hypotheses.values(),
            key=lambda h: h.order_index
        )
        
        dx_to_move = self.hypotheses[diagnosis_name]
        current_index = next(
            i for i, dx in enumerate(ordered_diagnoses)
            if dx.diagnosis.name == diagnosis_name
        )

        ordered_diagnoses.pop(current_index)
        ordered_diagnoses.insert(new_index, dx_to_move)
        
        for i, dx in enumerate(ordered_diagnoses):
            dx.order_index = i
            self.hypotheses[dx.diagnosis.name] = dx
                
        self.logger.info(f"Updated order for {diagnosis_name} to {new_index}")

        if hasattr(st.session_state, 'saved_diagnoses'):
            st.session_state.saved_diagnoses = self.save_state()

    def remove_diagnosis(self, diagnosis_name: str):
        """Remove a diagnosis from the differential."""
        if diagnosis_name in self.hypotheses:
            del self.hypotheses[diagnosis_name]
            self.logger.info(f"Removed diagnosis: {diagnosis_name}")

    def get_ranked_differential(self) -> List[Diagnosis]:
        """Get current differential diagnoses ranked by user order."""
        sorted_hypotheses = sorted(
            self.hypotheses.values(),
            key=lambda h: h.order_index
        )
        return [h.diagnosis for h in sorted_hypotheses]
    
    def swap_diagnoses(self, dx1_name: str, dx2_name: str):
        """Swap the positions of two diagnoses."""
        if dx1_name not in self.hypotheses or dx2_name not in self.hypotheses:
            return
            
        dx1 = self.hypotheses[dx1_name]
        dx2 = self.hypotheses[dx2_name]
        
        dx1.order_index, dx2.order_index = dx2.order_index, dx1.order_index
        
        self.hypotheses[dx1_name] = dx1
        self.hypotheses[dx2_name] = dx2
        
        if hasattr(st.session_state, 'saved_diagnoses'):
            st.session_state.saved_diagnoses = self.save_state()

    def save_state(self) -> List[Dict]:
        """Export current diagnoses for state persistence"""
        return [{
            "name": h.diagnosis.name,
            "category": h.diagnosis.category,
            "notes": h.notes,
            "order_index": h.order_index
        } for h in self.hypotheses.values()]

    def restore_diagnosis(self, dx_data: Dict):
        """Restore a diagnosis from saved state"""
        name = dx_data["name"]
        if name not in self.hypotheses:
            diagnosis = Diagnosis(
                name=name,
                category=dx_data["category"],
                key_features=[],
                supporting_evidence=[],
                refuting_evidence=[]
            )
            hypothesis = DiagnosticHypothesis(
                diagnosis=diagnosis,
                proposed_at=datetime.now(),
                notes=dx_data["notes"],
                order_index=dx_data["order_index"]
            )
            self.hypotheses[name] = hypothesis

