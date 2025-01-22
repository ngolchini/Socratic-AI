# Clinical AI Tutor - Application Structure Overview

## Core Application Structure

The application is built using Streamlit and follows a modular architecture with clear separation of concerns. Here's the high-level organization:

### Main Components

- `app.py` - Main application entry point and orchestrator
- `managers/` - Core business logic managers
- `models/` - Data models and type definitions
- `prompts/` - Configuration files for different clinical phases
- `cases/` - Clinical case data files

## Key Managers

1. **ClinicalCaseTutor** (app.py)
   - Main application class
   - Initializes and coordinates all other managers
   - Manages session state and UI flow

2. **CaseManager** (case_manager.py)
   - Loads and manages clinical case data
   - Handles case state tracking
   - Provides case metadata and phase configurations

3. **DifferentialManager** (differential_manager.py)
   - Manages differential diagnoses
   - Tracks diagnostic hypotheses and evidence
   - Handles diagnosis ranking and updates

4. **DisplayManager** (display_manager.py)
   - Controls UI layout and components
   - Manages chat interface
   - Handles differential diagnosis display
   - Updates case information panels

5. **PhaseManager** (phase_manager.py)
   - Controls progression through clinical phases
   - Assesses topic relevance and coverage
   - Manages phase transitions
   - Generates phase summaries

6. **PromptManager** (prompt_manager.py)
   - Constructs system prompts
   - Manages teaching points
   - Handles topic redirections
   - Provides phase-specific guidance

7. **LLMManager** (llm_manager.py)
   - Manages OpenAI API interactions
   - Handles different response types (conversation, JSON)
   - Maintains conversation context

## Data Models

1. **Phase Models** (phase.py)
   - `PhaseType`: Enum for different clinical phases
   - `TeachingPoint`: Educational objectives
   - `ClinicalElement`: Clinical information elements
   - `PhaseConfig`: Phase-specific configurations

2. **Case Models** (case.py)
   - `CaseData`: Complete case representation
   - `CaseMetadata`: Case information
   - `Diagnosis`: Diagnostic information
   - `DiagnosisCategory`: Classification of diagnoses

3. **Assessment Models** (assessment.py)
   - `TopicAssessment`: Evaluates response relevance
   - `CoverageAssessment`: Tracks clinical element coverage
   - `TopicRelevance`: Classification of topic relevance
   - `RedirectType`: Types of topic redirections

## Configuration Files

### Prompt Configurations (prompts/)
- `base.json`: Core teaching instructions
- `phases/`: Phase-specific configurations
  - `history.json`
  - `physical.json`
  - `testing.json`
  - `management.json`
  - `discussion.json`

## Session State Management

The application maintains session state for:
- Current case data
- Chat messages
- Phase progress
- Differential diagnoses
- Assessment cache
- Phase summaries

## Clinical Phase Flow

1. **History Phase**
   - Initial patient presentation
   - Systematic history taking
   - Initial differential formation

2. **Physical Phase**
   - Systematic examination
   - Finding interpretation
   - Differential refinement

3. **Testing Phase**
   - Test selection
   - Result interpretation
   - Diagnosis confirmation

4. **Management Phase**
   - Treatment planning
   - Patient education
   - Follow-up planning

5. **Discussion Phase**
   - Case synthesis
   - Learning point review
   - Clinical reasoning assessment

## Key Features

1. **Adaptive Learning**
   - Topic relevance assessment
   - Coverage tracking
   - Dynamic teaching point delivery

2. **Differential Diagnosis**
   - Real-time hypothesis tracking
   - Evidence strength assessment
   - User-modifiable rankings

3. **Phase Management**
   - Systematic progression
   - Completion criteria checking
   - Summary generation

4. **UI Components**
   - Chat interface
   - Case information panel
   - Differential diagnosis panel
   - Phase progress tracker