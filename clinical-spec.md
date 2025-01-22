# Clinical Case Tutor Technical Specification

## System Overview

The Clinical Case Tutor is an interactive educational platform that facilitates Socratic-style medical case discussions between AI tutors and medical trainees. The system guides learners through standardized clinical cases using carefully controlled phase progression and diagnostic reasoning tracking.

## Directory Structure

```
clinical-case-tutor/
├── app.py                             # Main Streamlit application
├── requirements.txt                    # Project dependencies
├── managers/
│   ├── phase_manager.py               # Phase progression control
│   ├── prompt_manager.py              # System prompt construction
│   ├── display_manager.py             # UI component management
│   ├── case_manager.py                # Case data handling
│   └── differential_manager.py         # Diagnostic reasoning tracking
├── models/
│   ├── phase.py                       # Phase and teaching models
│   ├── case.py                        # Case and diagnosis models
│   └── assessment.py                  # Assessment models
├── prompts/                           # Prompt templates
│   ├── base.json                      # Base system instructions
│   ├── phases/                        # Phase-specific prompts
│   └── redirects.json                 # Redirection templates
├── cases/                             # Case data files
└── utils/                             # Utility functions
```

## Core Components

### Models

The system uses strongly-typed data models to maintain consistency throughout the application:

1. Phase Models
- PhaseType enumeration for different discussion phases
- TeachingPoint for tracking learning objectives
- ClinicalElement for trackable clinical information
- Phase for complete phase configuration and state

2. Case Models
- CaseData for complete case representation
- CaseMetadata for case organization
- Diagnosis for differential diagnosis entries

3. Assessment Models
- TopicAssessment for evaluating discussion relevance
- CoverageAssessment for tracking teaching progress

### PhaseManager

The PhaseManager controls the progression through case phases while maintaining educational integrity. Key responsibilities include:

1. Phase State Management
- Tracking current phase and completion status
- Monitoring teaching point coverage
- Validating phase transitions

2. Topic Control
- Assessing topic relevance through focused LLM calls
- Generating appropriate redirections
- Tracking prohibited topics

3. Element Coverage
- Monitoring required vs optional elements
- Tracking teaching point delivery
- Generating phase summaries

### PromptManager

The PromptManager constructs appropriate system prompts throughout the case discussion. Key features include:

1. Dynamic Prompt Construction
- Base Socratic instruction integration
- Phase-specific guidance inclusion
- Context-aware prompt building

2. Redirection Management
- Multiple levels of redirection (gentle to direct)
- Educational context preservation
- Phase-appropriate guidance

### DisplayManager

The DisplayManager creates an intuitive Streamlit interface that balances educational content with supporting information. Key components include:

1. Layout Management
- Chat interface for primary discussion
- Case information panel
- Differential diagnosis tracking
- Optional debug interface

2. Component Updates
- Real-time chat updates
- Phase progress indicators
- Clinical information display
- Teaching point presentation

### CaseManager

The CaseManager handles case data and maintains case state throughout the educational session. Key responsibilities include:

1. Case Data Management
- Loading and validating case files
- Maintaining case state
- Tracking session progress

2. Element Tracking
- Recording covered elements
- Managing teaching point delivery
- Generating phase summaries

### DifferentialManager

The DifferentialManager tracks the evolution of diagnostic reasoning throughout the case. Key features include:

1. Hypothesis Management
- Tracking diagnostic hypotheses
- Managing evidence strength
- Calculating likelihood scores

2. Evidence Assessment
- LLM-based evidence evaluation
- Evidence strength thresholding
- Supporting/refuting evidence tracking

3. Reasoning Documentation
- Evidence trail maintenance
- Diagnostic evolution tracking
- Reasoning summary generation

## Implementation Details

### Phase Control System

The phase control system ensures proper progression while maintaining educational value:

1. Topic Assessment
- LLM analysis of user messages
- Prohibited topic detection
- Phase-appropriate guidance

2. Coverage Tracking
- Required element monitoring
- Teaching point delivery
- Critical information tracking

3. Phase Advancement
- Completion criteria validation
- Phase summary generation
- Context preservation

### Educational Control

The system maintains educational integrity through:

1. Socratic Method
- Probing question generation
- Guided discovery
- Structured feedback

2. Topic Management
- Phase-appropriate discussion
- Gentle redirection
- Teaching point integration

### User Interface

The interface provides a clean, educational experience through:

1. Primary Components
- Threaded chat interface
- Phase progress tracking
- Differential diagnosis panel

2. Supporting Features
- Case information display
- Debug mode toggle
- Teaching point presentation

### Data Flow

The system maintains consistent data flow between components:

1. Phase Progression
- PhaseManager controls state
- PromptManager generates instructions
- DisplayManager updates interface

2. Diagnostic Reasoning
- DifferentialManager tracks hypotheses
- CaseManager maintains context
- DisplayManager presents evolution

## Future Considerations

1. Enhanced Features
- Learning style adaptation
- Performance analytics
- Custom case creation

2. Technical Improvements
- Optimized LLM calls
- Enhanced evidence assessment
- Advanced analytics

3. Educational Enhancements
- Additional teaching modalities
- Collaborative learning features
- Extended feedback mechanisms

## Development Guidelines

1. Code Structure
- Maintain strong typing
- Use descriptive naming
- Implement comprehensive documentation

2. LLM Integration
- Use focused, efficient calls
- Implement robust error handling
- Maintain context appropriately

3. Interface Development
- Prioritize educational clarity
- Maintain consistent design
- Implement responsive updates

This specification provides a comprehensive framework for the Clinical Case Tutor system, emphasizing educational value through careful phase management, sophisticated diagnostic tracking, and intuitive interface design.