import pandas as pd
import json
import os
from pathlib import Path
import logging
import tiktoken
from openai import OpenAI
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_tokens(input_strings, encoding_name="cl100k_base"):
    """Count tokens in a list of strings."""
    tokenizer = tiktoken.get_encoding(encoding_name)
    
    token_counts = []
    for text in input_strings:
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))
    return token_counts

def save_case_json(response_content, output_path):
    """Save JSON content to file."""
    try:
        structured_case = json.loads(response_content)
        
        # Save to file with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_case, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully saved case to {output_path}")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Error: Response is not valid JSON: {e}")
        logger.error(f"Raw response content: {response_content[:500]}...")
        return False
        
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        return False

def create_ddx_prompt(section, case_data):
    """Create differential diagnosis prompt for each phase."""
    # Build the case information text based on section
    case_info = f"Original Presentation: {case_data['metadata']['original_presentation']}\n\n"
    
    sections_to_include = []
    if section == "history" or section == "physical" or section == "testing":
        sections_to_include.append("history")
    if section == "physical" or section == "testing":
        sections_to_include.append("physical")
    if section == "testing":
        sections_to_include.append("testing")
    
    for curr_section in sections_to_include:
        case_info += f"{curr_section.upper()}:\n"
        # Add required elements
        for item in case_data['clinical_elements'][curr_section]['required']:
            case_info += f"Question: {item['content']}\n"
            case_info += f"Finding: {item['response']}\n\n"
        # Add optional elements
        if 'optional' in case_data['clinical_elements'][curr_section]:
            for item in case_data['clinical_elements'][curr_section]['optional']:
                if item.get('elicited', False):
                    case_info += f"Question: {item['content']}\n"
                    case_info += f"Finding: {item['response']}\n\n"

    messages = [
        {
            "role": "system",
            "content": """You are an expert medical diagnostician. Your task is to generate a differential diagnosis 
            based on the available case information. Consider all information provided cumulatively up to this point.
            
            For each diagnosis in the differential:
            1. Assess its likelihood given the current information
            2. Identify key features that would be expected
            3. List supporting and refuting evidence from the case
            4. Specify what additional information would be helpful
            
            Format your response as a JSON array matching this structure:
            {
                "current_ideal_differential_diagnosis": [
                    {
                        "name": "diagnosis name",
                        "category": "disease category",
                        "likelihood": "high/medium/low based on current information",
                        "key_features": ["expected feature 1", "expected feature 2"],
                        "supporting_evidence": ["evidence from case that supports this diagnosis"],
                        "refuting_evidence": ["evidence from case that refutes this diagnosis"],
                        "additional_information_needed": ["specific questions or tests needed"]
                    }
                ]
            }
            
            Important: Return ONLY the JSON object with no additional text."""
        },
        {
            "role": "user",
            "content": f"""Based on the following case information up to this point, 
            generate a differential diagnosis list that reflects what an expert clinician 
            should be considering at this stage:\n\n{case_info}"""
        }
    ]
    
    return messages

def process_case_from_csv_row(client, row, case_schema, index, output_dir):
    """Process a single row from CSV into a case."""
    logger.info(f"Processing case #{index}")
    
    # Ensure row is a dictionary with keys for CSV columns
    row_dict = dict(row)
    
    # Print available columns for debugging
    logger.info(f"Available columns for case #{index}: {list(row_dict.keys())}")
    
    # Extract case text using correct columns from your CSV
    # Try different columns that may contain the case text
    case_text = None
    for column in ['PoC', 'PD', 'CD', 'DDx']:
        if column in row_dict and isinstance(row_dict[column], str) and len(row_dict[column]) > 100:
            case_text = row_dict[column]
            logger.info(f"Using column '{column}' for case text (length: {len(case_text)})")
            break
    
    # Get title from appropriate column
    case_title = row_dict.get('title', f'Case #{index}')
    
    if not case_text:
        # Print first few characters of each column to help debug
        for col, val in row_dict.items():
            if isinstance(val, str):
                preview = val[:50] + "..." if len(val) > 50 else val
                logger.info(f"Column '{col}' content preview: {preview}")
        logger.error(f"No suitable case text found for case #{index}")
        return None
    
    logger.info(f"Processing case: {case_title} (#{index}) with text length: {len(case_text)}")

    
    # Define schema as a separate variable - not in the f-string
    schema_example = '''
    {
        "metadata": {
            "id": "case_ID",
            "title": "Case title",
            "original_presentation": "First few sentences introducing the case",
            "difficulty": "intermediate",
            "specialties": ["relevant specialties"],
            "keywords": ["relevant keywords"],
            "source": "source of the case",
            "version": "1.0"
        },
        "clinical_elements": {
            "history": {
                "required": [
                    {
                        "id": "H1",
                        "content": "Question about history",
                        "response": "History information from case",
                        "teaching_points": [
                            {
                                "id": "TP1",
                                "content": "Teaching point about this history element",
                                "covered": false
                            }
                        ],
                        "elicited": false
                    }
                ],
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
            "name": "final diagnosis if known",
            "category": "category",
            "key_features": [],
            "supporting_evidence": [],
            "refuting_evidence": []
        }
    }
    '''
    
    # Structure the case
    system_message = {
        "role": "system",
        "content": f"""You are a medical case structuring assistant. Your task is to convert unstructured clinical case text into a structured JSON format. 
        Follow these guidelines:
        1. Extract the "original presentation" - the first few sentences that introduce the case
        2. Identify history elements (patient history, symptoms, etc.)
        3. Identify physical examination findings
        4. Identify test results and diagnostic procedures
        5. Identify management decisions and treatments
        6. Create teaching points for each clinical element

        Format each element with:
        - id: A unique code (H1, P1, T1, M1, etc. for section and number)
        - content: A question that would elicit this information
        - response: The actual information from the case text
        - teaching_points: Educational points related to this information

        Return a valid JSON object matching this structure:
        
        {schema_example}
        
        Important: Return ONLY the JSON object with no additional text or explanation."""
    }

    user_message = {
        "role": "user",
        "content": f"Please structure this clinical case according to the schema shown above:\n\n{case_text}"
    }

    try:
        # Generate structured case
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[system_message, user_message],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        structured_case_json = response.choices[0].message.content
        structured_case = json.loads(structured_case_json)
        
        # Override case ID and add other metadata from the CSV
        case_id = f"case_{index:03d}"
        structured_case['metadata']['id'] = case_id
        structured_case['metadata']['title'] = case_title
        
        # Add any additional metadata from CSV
        if 'organ_system' in row_dict:
            specialties = row_dict['organ_system'].split(',')
            structured_case['metadata']['specialties'] = [s.strip() for s in specialties if s.strip()]
        
        if 'keywords' in row_dict:
            keywords = row_dict['keywords'].split(',')
            structured_case['metadata']['keywords'] = [k.strip() for k in keywords if k.strip()]
        
        if 'diagnosis' in row_dict:
            structured_case['final_diagnosis']['name'] = row_dict['diagnosis']
        
        # Create output file path
        case_file_path = os.path.join(output_dir, f"{case_id}.json")
        
        # Now add differential diagnoses for each phase
        for section in ["history", "physical", "testing"]:
            try:
                messages = create_ddx_prompt(section, structured_case)
                ddx_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                ddx = json.loads(ddx_response.choices[0].message.content)
                structured_case['clinical_elements'][section]['current_ideal_differential_diagnosis'] = \
                    ddx['current_ideal_differential_diagnosis']
                
                logger.info(f"Added differential for {section} phase")
            except Exception as e:
                logger.error(f"Error adding differential for {section}: {str(e)}")
        
        # Save the final case
        if save_case_json(json.dumps(structured_case), case_file_path):
            logger.info(f"Successfully saved case {case_id}")
            return case_id
        else:
            logger.error(f"Failed to save case {case_id}")
            return None
        
    except Exception as e:
        logger.error(f"Error processing case #{index}: {str(e)}")
        return None

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python process_cases.py <api_key> [start_index=1] [end_index=50] [csv_path=path/to/csv]")
        sys.exit(1)
    
    api_key = sys.argv[1]
    start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end_index = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    csv_path = sys.argv[4] if len(sys.argv) > 4 else "/Users/nilogolchini/Desktop/socratic_ai_tz_updates/NEJM_CPC_all_section_04_10_23.csv"
    
    logger.info(f"Starting case processing with parameters:")
    logger.info(f"- API Key: {'*' * 5}{api_key[-4:] if len(api_key) > 4 else ''}")
    logger.info(f"- Start index: {start_index}")
    logger.info(f"- End index: {end_index}")
    logger.info(f"- CSV Path: {csv_path}")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Create output directory for cases
    output_dir = Path("cases")
    output_dir.mkdir(exist_ok=True)
    
    # Load CSV data
    try:
        logger.info(f"Loading CSV from: {csv_path}")
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            print(f"Error: CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded CSV with {len(df)} rows")
        
        # Print detailed info about the first row
        logger.info("CSV columns: " + ", ".join(df.columns))
        
        # Sample the first row to understand structure
        first_row = df.iloc[0].to_dict()
        logger.info("First row contents:")
        for key, value in first_row.items():
            if isinstance(value, str):
                val_preview = f"{value[:100]}..." if len(value) > 100 else value
                logger.info(f"  {key} ({len(value)} chars): {val_preview}")
            else:
                logger.info(f"  {key} (non-string): {value}")
        
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        print(f"Error loading CSV file: {str(e)}")
        return
    
    # Calculate indices for slicing
    start_idx = max(0, start_index - 1)  # Convert to 0-based indexing
    end_idx = min(len(df), end_index)    # Ensure we don't go beyond dataframe length
    total_to_process = end_idx - start_idx
    
    # Process the specified range of cases
    processed_cases = []
    for i, (_, row) in enumerate(df.iloc[start_idx:end_idx].iterrows(), start=start_index):
        logger.info(f"Processing case {i}/{end_index} ({i-start_index+1}/{total_to_process})")
        case_id = process_case_from_csv_row(client, row, None, i, str(output_dir))
        if case_id:
            processed_cases.append(case_id)
    
    logger.info(f"Processing complete. {len(processed_cases)} cases processed out of {total_to_process} requested.")
    print(f"Processing complete. {len(processed_cases)} cases processed out of {total_to_process} requested.")

if __name__ == "__main__":
    main()