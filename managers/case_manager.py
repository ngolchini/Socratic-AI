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
            
            # Store raw data for accessing ideal differential later
            raw_data = case_json
            
            # Load metadata
            metadata = CaseMetadata(
                id=case_json['metadata']['id'],
                title=case_json['metadata'].get('original_presentation', case_json['metadata'].get('title', '')),
                difficulty=case_json['metadata'].get('difficulty', 'intermediate'),
                specialties=case_json['metadata'].get('specialties', []),
                keywords=case_json['metadata'].get('keywords', [])
            )
            
            # Load phases with config injection
            phases = {}
            for phase_type in PhaseType:
                phase_data = case_json['clinical_elements'].get(phase_type.value, {})
                config = self.get_phase_config(phase_type)
                
                # Create Phase object
                phase = Phase.from_json(phase_data, config)
                
                # Store ideal differential if available
                current_ideal_ddx = phase_data.get('current_ideal_differential_diagnosis', [])
                if current_ideal_ddx:
                    phase.current_ideal_differential_diagnosis = current_ideal_ddx
                    
                phases[phase_type] = phase
            
            # Load differential diagnoses
            differential_diagnosis = []
            for dx_data in case_json.get('differential_diagnosis', []):
                if dx_data:  # Check if valid data exists
                    diagnosis = self._construct_diagnosis(dx_data)
                    differential_diagnosis.append(diagnosis)
            
            # Load final diagnosis
            final_diagnosis = None
            if 'final_diagnosis' in case_json and case_json['final_diagnosis']:
                final_diagnosis = self._construct_diagnosis(case_json['final_diagnosis'])
            else:
                # Create a placeholder diagnosis if none exists
                final_diagnosis = Diagnosis(
                    name="Diagnosis Pending",
                    category=DiagnosisCategory.POSSIBLE,
                    key_features=[]
                )
            
            # Create the CaseData object
            case_data = CaseData(
                metadata=metadata,
                phases=phases,
                differential_diagnosis=differential_diagnosis,
                final_diagnosis=final_diagnosis
            )
            
            # Store raw data for future reference
            case_data._raw_data = raw_data
            
            self.logger.info(f"Successfully loaded case {case_id}")
            return case_data
            
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
    def search_cases(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for cases matching a natural language query.
        
        Args:
            query: Natural language search query
            top_k: Number of top results to return
            
        Returns:
            List of matching case metadata, sorted by relevance
        """
        self.logger.info(f"Searching for cases matching: {query}")
        
        available_cases = [f.stem for f in self.cases_dir.glob("*.json")]
        case_metadata = []
        
        # Load metadata for all available cases
        for case_id in available_cases:
            try:
                case_path = self.cases_dir / f"{case_id}.json"
                with open(case_path, 'r') as f:
                    case_data = json.load(f)
                    case_metadata.append(case_data['metadata'])
            except Exception as e:
                self.logger.error(f"Error loading case {case_id}: {str(e)}")
        
        # If no cases found, return empty list
        if not case_metadata:
            return []
        
        # Implement search based on keyword matching
        results = []
        query_terms = query.lower().split()
        
        for metadata in case_metadata:
            score = 0
            # Search in title
            title = metadata.get('title', '').lower()
            for term in query_terms:
                if term in title:
                    score += 3
            
            # Search in presentation
            presentation = metadata.get('original_presentation', '').lower()
            for term in query_terms:
                if term in presentation:
                    score += 2
            
            # Search in keywords
            for keyword in metadata.get('keywords', []):
                keyword = keyword.lower().strip()
                for term in query_terms:
                    if term in keyword:
                        score += 5
                        break
            
            # Search in specialties
            for specialty in metadata.get('specialties', []):
                specialty = specialty.lower().strip()
                for term in query_terms:
                    if term in specialty:
                        score += 4
                        break
            
            # Add to results if any match
            if score > 0:
                results.append((metadata, score))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return [item[0] for item in results[:top_k]]

    def _rank_cases_with_llm(self, query: str, case_metadatas: List[Dict]) -> List[Dict]:
        """
        Use LLM to rank cases by relevance to the query.
        
        Args:
            query: Natural language search query
            case_metadatas: List of case metadata dictionaries
            
        Returns:
            List of case metadatas, sorted by relevance
        """
        system_prompt = """You are a clinical case search assistant. 
        Given a user's search query and a list of cases, rank the cases by relevance to the query.
        Consider the case title, presentation, specialties, and keywords in your ranking.
        Return a JSON array of case ids, ordered from most to least relevant.
        """
        
        # Format the cases in a way that's easy for the LLM to process
        cases_info = []
        for i, metadata in enumerate(case_metadatas):
            cases_info.append({
                "index": i,
                "id": metadata["id"],
                "title": metadata["title"],
                "presentation": metadata.get("original_presentation", ""),
                "specialties": metadata.get("specialties", []),
                "keywords": metadata.get("keywords", [])
            })
        
        user_message = f"Query: {query}\n\nCases:\n{json.dumps(cases_info, indent=2)}"
        
        try:
            # Use the LLM to rank the cases
            response = None
            if hasattr(self, 'llm_manager'):
                response = self.llm_manager.get_json_response(
                    system_prompt,
                    user_message
                )
            else:
                # Fallback to simple keyword matching if no LLM manager
                return self._fallback_case_ranking(query, case_metadatas)
            
            # Process the ranked IDs returned from the LLM
            ranked_ids = response.get("ranked_ids", [])
            
            # Map back to the original case metadata objects
            case_dict = {m["id"]: m for m in case_metadatas}
            ranked_cases = [case_dict[case_id] for case_id in ranked_ids if case_id in case_dict]
            
            # Add any remaining cases that weren't in the ranked list
            for metadata in case_metadatas:
                if metadata["id"] not in ranked_ids:
                    ranked_cases.append(metadata)
            
            return ranked_cases
        
        except Exception as e:
            self.logger.error(f"Error ranking cases: {str(e)}")
            return self._fallback_case_ranking(query, case_metadatas)

    def _fallback_case_ranking(self, query: str, case_metadatas: List[Dict]) -> List[Dict]:
        """
        Simple fallback ranking mechanism based on keyword matching.
        
        Args:
            query: Natural language search query
            case_metadatas: List of case metadata dictionaries
            
        Returns:
            List of case metadatas, sorted by relevance
        """
        query_terms = query.lower().split()
        ranked_cases = []
        
        for metadata in case_metadatas:
            score = 0
            
            # Check title
            title = metadata.get("title", "").lower()
            for term in query_terms:
                if term in title:
                    score += 3
            
            # Check presentation
            presentation = metadata.get("original_presentation", "").lower()
            for term in query_terms:
                if term in presentation:
                    score += 2
            
            # Check keywords
            keywords = [k.lower() for k in metadata.get("keywords", [])]
            for term in query_terms:
                if term in keywords:
                    score += 5
            
            # Check specialties
            specialties = [s.lower() for s in metadata.get("specialties", [])]
            for term in query_terms:
                if term in specialties:
                    score += 4
            
            ranked_cases.append((metadata, score))
        
        # Sort by score, descending
        ranked_cases.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the metadata
        return [rc[0] for rc in ranked_cases]
    
