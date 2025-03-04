import csv
import json
import os
from pathlib import Path
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def import_cases_from_csv(csv_path: str, case_dir: str, limit: int = 50) -> List[str]:
    """
    Import cases from a CSV file and save them to the cases directory.
    
    Args:
        csv_path: Path to the CSV file containing case data
        case_dir: Directory to save the case files
        limit: Maximum number of cases to import (default: 50)
        
    Returns:
        List of case IDs that were successfully imported
    """
    case_dir_path = Path(case_dir)
    case_dir_path.mkdir(exist_ok=True, parents=True)
    
    imported_cases = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                
                # Create case ID from index or an existing ID field if available
                case_id = f"case_{i+1:03d}" if 'id' not in row else row['id']
                
                # Format the case based on your schema
                case_data = format_case_from_csv_row(row, case_id)
                
                # Save the case to a JSON file
                case_file_path = case_dir_path / f"{case_id}.json"
                with open(case_file_path, 'w', encoding='utf-8') as f:
                    json.dump(case_data, f, indent=2)
                
                imported_cases.append(case_id)
                logger.info(f"Imported case {case_id}")
    
    except Exception as e:
        logger.error(f"Error importing cases from CSV: {str(e)}")
        raise
    
    return imported_cases

def format_case_from_csv_row(row: Dict[str, str], case_id: str) -> Dict[str, Any]:
    """
    Format a case from a CSV row according to the application's case schema.
    
    Args:
        row: CSV row data
        case_id: ID to assign to the case
        
    Returns:
        Formatted case data
    """
    # Map CSV columns to case schema - adjust according to your CSV structure
    title = row.get('title', 'Untitled Case')
    presentation = row.get('presentation', '')
    difficulty = row.get('difficulty', 'intermediate')
    specialties = row.get('specialties', '').split(',') if row.get('specialties') else []
    keywords = row.get('keywords', '').split(',') if row.get('keywords') else []
    
    # Create basic case structure
    case_data = {
        "metadata": {
            "id": case_id,
            "title": title,
            "original_presentation": presentation,
            "difficulty": difficulty,
            "specialties": specialties,
            "keywords": keywords,
            "source": row.get('source', ''),
            "version": "1.0"
        },
        "clinical_elements": {
            "history": {
                "required": [],
                "optional": []
            },
            "physical": {
                "required": [],
                "optional": []
            },
            "testing": {
                "required": [],
                "optional": []
            },
            "management": {
                "required": [],
                "optional": []
            },
            "discussion": {
                "required": [],
                "optional": []
            }
        },
        "differential_diagnosis": [],
        "final_diagnosis": {
            "name": row.get('final_diagnosis', 'Unknown'),
            "category": "Unspecified",
            "key_features": []
        }
    }
    
    return case_data