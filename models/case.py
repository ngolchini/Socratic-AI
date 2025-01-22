from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from .phase import Phase, PhaseType

class DiagnosisCategory(Enum):
    LIKELY = "likely"
    POSSIBLE = "possible"
    DONT_MISS = "dont_miss"

@dataclass
class Diagnosis:
    """Represents a diagnosis in the differential"""
    name: str
    category: DiagnosisCategory
    key_features: List[str]
    supporting_evidence: List[str] = None
    refuting_evidence: List[str] = None

    def __post_init__(self):
        if self.supporting_evidence is None:
            self.supporting_evidence = []
        if self.refuting_evidence is None:
            self.refuting_evidence = []

@dataclass
class CaseMetadata:
    """Metadata about the clinical case"""
    id: str
    title: str
    difficulty: str
    specialties: List[str]
    keywords: List[str]

@dataclass
class CaseData:
    """Complete representation of a clinical case"""
    metadata: CaseMetadata
    phases: Dict[PhaseType, Phase]
    differential_diagnosis: List[Diagnosis]
    final_diagnosis: Diagnosis

    def get_current_phase(self, phase_type: PhaseType) -> Phase:
        """Get the phase data for a specific phase"""
        phase = self.phases[phase_type]
        
        # Add the ideal differential from raw data if it exists
        if hasattr(self, '_raw_data'):
            phase_data = self._raw_data['clinical_elements'].get(phase_type.value.lower(), {})
            phase.current_ideal_differential_diagnosis = phase_data.get('current_ideal_differential_diagnosis', [])
        return phase
    
    def update_differential(self, diagnosis: Diagnosis):
        """Update or add a diagnosis to the differential"""
        existing = next(
            (d for d in self.differential_diagnosis if d.name == diagnosis.name), 
            None
        )
        if existing:
            # Update category if different
            if diagnosis.category != existing.category:
                existing.category = diagnosis.category
            # Add new evidence
            if diagnosis.supporting_evidence:
                for evidence in diagnosis.supporting_evidence:
                    if evidence not in existing.supporting_evidence:
                        existing.supporting_evidence.append(evidence)
            if diagnosis.refuting_evidence:
                for evidence in diagnosis.refuting_evidence:
                    if evidence not in existing.refuting_evidence:
                        existing.refuting_evidence.append(evidence)
        else:
            self.differential_diagnosis.append(diagnosis)