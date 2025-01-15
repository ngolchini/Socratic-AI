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

