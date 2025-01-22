from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class PhaseType(Enum):
    HISTORY = "history"
    PHYSICAL = "physical"
    TESTING = "testing"
    MANAGEMENT = "management"
    DISCUSSION = "discussion"

@dataclass
class TeachingPoint:
    id: str
    content: str
    covered: bool = False
    coverage_notes: Optional[str] = None

    @staticmethod
    def from_json(data: Dict) -> 'TeachingPoint':
        return TeachingPoint(
            id=data['id'],
            content=data['content'],
            covered=data.get('covered', False),
            coverage_notes=data.get('coverage_notes')
        )

@dataclass
class ClinicalElement:
    id: str
    content: str
    required: bool
    response: str
    teaching_points: List[TeachingPoint]
    elicited: bool = False
    elicited_content: Optional[str] = None

    @staticmethod
    def from_json(data: Dict) -> 'ClinicalElement':
        teaching_points = [
            TeachingPoint.from_json(tp) for tp in data.get('teaching_points', [])
        ]
        return ClinicalElement(
            id=data['id'],
            content=data['content'],
            required=data.get('required', True),
            response=data['response'],
            teaching_points=teaching_points,
            elicited=data.get('elicited', False),
            elicited_content=data.get('elicited_content')
        )

@dataclass
class PhaseConfig:
    opening_prompt: str
    completion_message: str
    prohibited_topics: List[str]
    advancement_criteria: List[str]

    @staticmethod
    def from_json(data: Dict) -> 'PhaseConfig':
        return PhaseConfig(
            opening_prompt=data.get('opening_prompt', ''),
            completion_message=data.get('completion_message', ''),
            prohibited_topics=data.get('prohibited_topics', []),
            advancement_criteria=data.get('advancement_criteria', [])
        )

@dataclass
class Phase:
    required_elements: List[ClinicalElement]
    optional_elements: List[ClinicalElement]
    teaching_points: List[TeachingPoint]
    config: PhaseConfig

    @staticmethod
    def from_json(data: Dict, config: Optional[PhaseConfig] = None) -> 'Phase':
        """Create Phase object with proper config handling."""
        # Convert required elements
        required_elements = [
            ClinicalElement.from_json(elem) 
            for elem in data.get('required', [])
        ]
        
        # Convert optional elements
        optional_elements = [
            ClinicalElement.from_json(elem)
            for elem in data.get('optional', [])
        ]
        
        # Extract teaching points
        teaching_points = []
        for elem in required_elements + optional_elements:
            teaching_points.extend(elem.teaching_points)
        
        # Use provided config or create default
        phase_config = config or PhaseConfig(
            opening_prompt="What would you like to know about the patient?",
            completion_message="You've completed this phase.",
            prohibited_topics=[],
            advancement_criteria=[]
        )
        
        return Phase(
            required_elements=required_elements,
            optional_elements=optional_elements,
            teaching_points=teaching_points,
            config=phase_config
        )

    def is_complete(self) -> bool:
        """Check if all required elements have been elicited."""
        return all(elem.elicited for elem in self.required_elements)

    def get_uncovered_elements(self) -> List[ClinicalElement]:
        """Get list of elements that haven't been covered yet."""
        return [
            elem for elem in self.required_elements + self.optional_elements 
            if not elem.elicited
        ]