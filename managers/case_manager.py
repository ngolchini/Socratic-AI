from pathlib import Path
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

from models.case import CaseData, CaseMetadata, Diagnosis, DiagnosisCategory
from models.phase import Phase, PhaseType, ClinicalElement, TeachingPoint, PhaseConfig
from managers.prompt_manager import PromptManager  # Use relative import


class CaseManager:
    def __init__(self, cases_directory: str = "cases"):
        self.current_case: Optional[CaseData] = None
        self.cases_dir = Path(cases_directory)
        self.prompts_dir = Path("prompts/phases")
        self.prompt_manager = PromptManager()
        self.case_state: Dict = {}
        self.session_history: List[Dict] = []
        self.logger = logging.getLogger(__name__)
        self.phase_configs = {}

    def get_phase_config(self, phase_type: PhaseType) -> PhaseConfig:
        """Get phase configuration, loading from cache if available."""
        # Always reload config for new phases
        config_path = self.prompts_dir / f"{phase_type.value}.json"
        try:
            with open(config_path, "r") as f:
                config_json = json.load(f)
                self.phase_configs[phase_type] = PhaseConfig(
                    opening_prompt=config_json["opening_prompt"],
                    completion_message=config_json["completion_message"],
                    prohibited_topics=config_json["prohibited_topics"],
                    advancement_criteria=config_json["advancement_criteria"]
                )
        except Exception as e:
            self.logger.error(f"Error loading phase config for {phase_type}: {str(e)}")
            # Return default config if loading fails
            self.phase_configs[phase_type] = PhaseConfig(
                opening_prompt="What would you like to know about the patient?",
                completion_message="Phase complete.",
                prohibited_topics=[],
                advancement_criteria=[]
            )
        return self.phase_configs[phase_type]
    
    def load_case(self, case_id: str) -> CaseData:
        """Load a case from JSON file with proper config injection."""
        case_path = self.cases_dir / f"{case_id}.json"
        
        try:
            with open(case_path, 'r') as f:
                case_json = json.load(f)
            
            # Load metadata
            metadata = CaseMetadata(
                id=case_json['metadata']['id'],
                title=case_json['metadata']['title'],
                difficulty=case_json['metadata']['difficulty'],
                specialties=case_json['metadata']['specialties'],
                keywords=case_json['metadata']['keywords']
            )
            
            # Load phases with config injection
            phases = {}
            for phase_type in PhaseType:
                phase_data = case_json['clinical_elements'].get(phase_type.value, {})
                config = self.get_phase_config(phase_type)
                phases[phase_type] = Phase.from_json(phase_data, config)
            
            # Rest of the case loading logic...
            return CaseData(
                metadata=metadata,
                phases=phases,
                differential_diagnosis=[...],
                final_diagnosis=[...]
            )
        except Exception as e:
            self.logger.error(f"Error loading case {case_id}: {str(e)}")
            raise

    def _construct_case_data(self, case_json: Dict) -> CaseData:
        """Convert raw JSON data into structured CaseData object."""
        self.logger.debug("Beginning case data construction")
        
        metadata = self._construct_metadata(case_json["metadata"])
        phases = self._construct_phases(case_json["clinical_elements"])
        differential = [
            self._construct_diagnosis(dx)
            for dx in case_json["differential_diagnosis"]
        ]
        final_dx = self._construct_diagnosis(case_json["final_diagnosis"])
        
        self.logger.debug(f"Constructed phases: {phases.keys()}")
        
        return CaseData(
            metadata=metadata,
            phases=phases,
            differential_diagnosis=differential,
            final_diagnosis=final_dx
        )

    def _construct_metadata(self, metadata_json: Dict) -> CaseMetadata:
        """Construct case metadata object."""
        return CaseMetadata(
            id=metadata_json["id"],
            title=metadata_json["title"],
            difficulty=metadata_json["difficulty"],
            specialties=metadata_json["specialties"],
            keywords=metadata_json["keywords"]
        )

    def _construct_phases(self, elements_json: Dict) -> Dict[PhaseType, Phase]:
        """Construct phase objects using clinical elements."""
        phases = {}
        
        for phase_type in PhaseType:
            phase_elements = elements_json.get(phase_type.value, {})
            config = self.prompt_manager.get_phase_config(phase_type)
            
            phases[phase_type] = Phase(
                type=phase_type,
                config=config,  # Use config from PromptManager
                required_elements=[
                    self._construct_clinical_element(element)
                    for element in phase_elements.get("required", [])
                ],
                optional_elements=[
                    self._construct_clinical_element(element)
                    for element in phase_elements.get("optional", [])
                ],
                teaching_points=self._load_teaching_points(phase_type)
            )
            
        return phases

    def _load_phase_config(self, phase_type: PhaseType) -> PhaseConfig:
        """Load phase configuration from phase-specific JSON file."""
        if phase_type not in self.phase_configs:
            config_path = Path("prompts/phases") / f"{phase_type.value}.json"
            try:
                with open(config_path, "r") as f:
                    config_json = json.load(f)
                    self.phase_configs[phase_type] = PhaseConfig(
                        opening_prompt=config_json["opening_prompt"],
                        completion_message=config_json["completion_message"],
                        prohibited_topics=config_json["prohibited_topics"],
                        advancement_criteria=config_json["advancement_criteria"]
                    )
            except Exception as e:
                self.logger.error(f"Error loading phase config for {phase_type}: {str(e)}")
                self.phase_configs[phase_type] = PhaseConfig(
                    opening_prompt="What would you like to know about the patient?",
                    completion_message="Phase complete.",
                    prohibited_topics=[],
                    advancement_criteria=[]
                )
        return self.phase_configs[phase_type]

    def _load_teaching_points(self, phase_type: PhaseType) -> List[TeachingPoint]:
        """Load teaching points using PromptManager's loaded data."""
        phase_json = self.prompt_manager.phase_instructions[phase_type]
        return [
            TeachingPoint(
                id=point["id"],
                content=point["content"],
                required=point.get("required", True)
            )
            for point in phase_json.get("teaching_points", [])
        ]

    def _construct_clinical_element(self, element_data: Dict) -> ClinicalElement:
        """Convert raw element data into ClinicalElement object."""
        self.logger.info(f"Constructing clinical element: {element_data['id']}")
        element = ClinicalElement(
            id=element_data["id"],
            content=element_data["content"],
            required=element_data["required"],
            teaching_points=[
                TeachingPoint(
                    id=point["id"],
                    content=point["content"],
                    required=point.get("required", True)
                )
                for point in element_data.get("teaching_points", [])
            ],
            elicited=False,  # Explicitly set to False
            elicited_content=None
        )
        self.logger.info(f"Created element {element.id} with elicited={element.elicited}")
        return element

    def _construct_diagnosis(self, dx_data: Dict) -> Diagnosis:
        """Convert raw diagnosis data into Diagnosis object."""
        return Diagnosis(
            name=dx_data["name"],
            category=DiagnosisCategory(dx_data["category"]),
            likelihood_score=dx_data.get("likelihood_score", 0.0),
            key_features=dx_data["key_features"],
            supporting_evidence=dx_data.get("supporting_evidence", []),
            refuting_evidence=dx_data.get("refuting_evidence", [])
        )

    def _initialize_case_state(self):
        """Initialize the tracking state for the current case session."""
        self.case_state = {
            "start_time": datetime.now(),
            "current_phase": PhaseType.HISTORY,
            "covered_elements": set(),
            "teaching_points_covered": set(),
            "phase_summaries": {},
            "differential_updates": []
        }
        
        self.session_history.append({
            "timestamp": datetime.now(),
            "type": "case_start",
            "case_id": self.current_case.metadata.id
        })
        
        self.logger.info(f"Initialized case state for {self.current_case.metadata.id}")        

    def get_case_progress(self) -> Dict:
        """Get current progress through the case."""
        current_phase = self.case_state["current_phase"]
        phase = self.current_case.phases[current_phase]
        
        return {
            "current_phase": current_phase.value,
            "elements_covered": len(self.case_state["covered_elements"]),
            "teaching_points_covered": len(self.case_state["teaching_points_covered"]),
            "phase_completion": len([e for e in phase.required_elements if e.elicited]) / 
                              len(phase.required_elements) if phase.required_elements else 1.0,
            "time_elapsed": (datetime.now() - self.case_state["start_time"]).total_seconds()
        }

    def export_session_data(self) -> Dict:
        """Export complete session data for analysis or storage."""
        return {
            "case_metadata": self.current_case.metadata.__dict__,
            "session_history": self.session_history,
            "phase_summaries": self.case_state["phase_summaries"],
            "completion_metrics": self.get_case_progress(),
            "differential_updates": self.case_state["differential_updates"]
        }