import os
import json
import pandas as pd
import numpy as np
import faiss
import pickle
import time
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from openai import OpenAI

class JSONCaseEmbedder:
    """Creates embeddings and FAISS indexes for clinical cases stored in JSON files."""
    
    def __init__(self, api_key: str, embedding_model: str = 'text-embedding-3-large'):
        """Initialize the embedder with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model
        self.cases_data = []
        
        # Create output directories
        os.makedirs('data_files', exist_ok=True)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """Generate embeddings for a list of texts in batches."""
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Skip empty texts
            batch_texts = [text if text and text.strip() else "Empty content" for text in batch_texts]
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.embedding_model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Sleep to avoid rate limits
                time.sleep(0.5)
            except Exception as e:
                print(f"Error generating embeddings for batch {i}: {str(e)}")
                # Return empty embeddings for failed batch
                all_embeddings.extend([[0] * 1536] * len(batch_texts))
        
        return all_embeddings
    
    def load_json_cases(self, cases_dir: str) -> pd.DataFrame:
        """Load all JSON case files from directory."""
        all_cases = []
        
        for filename in tqdm(os.listdir(cases_dir), desc="Loading case files"):
            if filename.endswith('.json'):
                file_path = os.path.join(cases_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                    
                    # Extract case ID (from filename or metadata)
                    case_id = case_data.get("metadata", {}).get("id", os.path.splitext(filename)[0])
                    
                    # Extract all needed information
                    case_info = {
                        "case": case_id,
                        "title": case_data.get("metadata", {}).get("title", ""),
                        "one_line": case_data.get("metadata", {}).get("original_presentation", ""),
                        "specialties": ", ".join(case_data.get("metadata", {}).get("specialties", [])),
                        "keywords": ", ".join(case_data.get("metadata", {}).get("keywords", [])),
                        "difficulty": case_data.get("metadata", {}).get("difficulty", ""),
                    }
                    
                    # Extract patient of concern (PoC) from history elements
                    history_elements = []
                    for element in case_data.get("clinical_elements", {}).get("history", {}).get("required", []):
                        if "response" in element:
                            history_elements.append(element["response"])
                    case_info["PoC"] = " ".join(history_elements)
                    
                    # Extract differential diagnosis (DDx)
                    differential = []
                    for dx in case_data.get("clinical_elements", {}).get("history", {}).get("current_ideal_differential_diagnosis", []):
                        if "name" in dx:
                            differential.append(dx["name"])
                    case_info["DDx"] = ", ".join(differential)
                    
                    # Extract physical examination findings
                    physical_elements = []
                    for element in case_data.get("clinical_elements", {}).get("physical", {}).get("required", []):
                        if "response" in element:
                            physical_elements.append(element["response"])
                    case_info["PD"] = " ".join(physical_elements)
                    
                    # Extract testing findings
                    testing_elements = []
                    for element in case_data.get("clinical_elements", {}).get("testing", {}).get("required", []):
                        if "response" in element:
                            testing_elements.append(element["response"])
                    case_info["TD"] = " ".join(testing_elements)
                    
                    # Extract final diagnosis
                    final_dx = case_data.get("final_diagnosis", {})
                    case_info["FD"] = final_dx.get("name", "")
                    
                    # Extract URL (if exists)
                    case_info["url"] = ""  # Placeholder for URL if you have it
                    
                    all_cases.append(case_info)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        # Convert to DataFrame
        cases_df = pd.DataFrame(all_cases)
        print(f"Loaded {len(cases_df)} cases from {cases_dir}")
        
        # Save the full cases dataframe
        cases_df.to_csv("data_files/all_cases.csv", index=False)
        
        return cases_df
    
    def create_index_for_field(self, cases_df: pd.DataFrame, field_name: str) -> None:
        """Create a FAISS index for a specific field in the cases dataframe."""
        print(f"Creating index for {field_name}...")
        
        # Extract texts for the field
        texts = cases_df[field_name].astype(str).tolist()
        texts = [t if t != 'nan' and t != 'None' else '' for t in texts]
        
        # Get case labels
        labels = cases_df['case'].tolist()
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create DataFrame with labels and texts
        index_df = pd.DataFrame({
            'label': labels,
            'text': texts
        })
        
        # Create FAISS index
        vector_dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(vector_dimension)
        
        # Add vectors to index
        vectors = np.array(embeddings).astype('float32')
        index.add(vectors)
        
        # Save index and dataframe
        faiss.write_index(index, f"data_files/{field_name}_faiss.index")
        with open(f"data_files/{field_name.lower()}_df_ada3.pkl", 'wb') as f:
            pickle.dump(index_df, f)
        
        print(f"Index for {field_name} created and saved successfully.")
    
    def process_all_fields(self, cases_dir: str, fields: List[str]) -> None:
        """Process all specified fields from JSON case files."""
        # Load cases from JSON files
        cases_df = self.load_json_cases(cases_dir)
        
        # Process each field
        for field in fields:
            if field in cases_df.columns:
                self.create_index_for_field(cases_df, field)
            else:
                print(f"Warning: Field '{field}' not found in the cases data.")
        
        print("All fields processed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Create embeddings and FAISS indexes for JSON case files')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--cases_dir', type=str, default='cases', help='Directory containing JSON case files')
    parser.add_argument('--fields', type=str, nargs='+', 
                        default=['PoC', 'DDx', 'PD', 'TD', 'FD'], 
                        help='Fields to create indexes for')
    
    args = parser.parse_args()
    
    embedder = JSONCaseEmbedder(api_key=args.api_key)
    embedder.process_all_fields(args.cases_dir, args.fields)

if __name__ == "__main__":
    main()