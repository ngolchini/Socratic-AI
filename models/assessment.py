from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
from .phase import ClinicalElement, TeachingPoint

class TopicRelevance(Enum):
    ON_TOPIC = "on_topic"
    RELATED = "related"
    OFF_TOPIC = "off_topic"

class RedirectType(Enum):
    NONE = "none"
    GENTLE = "gentle"
    EDUCATIONAL = "educational"
    DIRECT = "direct"

@dataclass
class TopicAssessment:
    """Assessment of a user's response relevance"""
    relevance: TopicRelevance
    redirect_type: RedirectType
    #prohibited_topics: List[str]
    redirect_message: Optional[str] = None

@dataclass
class CoverageAssessment:
    """Assessment of teaching point and element coverage"""
    newly_covered_elements: List[ClinicalElement]
    newly_covered_points: List[TeachingPoint]
    missing_critical_elements: List[ClinicalElement]