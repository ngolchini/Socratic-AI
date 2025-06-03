import os
import re
import time
import json
import random
import pickle
import logging
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from email.message import EmailMessage
import smtplib

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import bcrypt
import yaml
import gspread
from dotenv import load_dotenv
from openai import AzureOpenAI
from google.oauth2.service_account import Credentials

from managers.case_manager import CaseManager
from managers.differential_manager import DifferentialManager
from managers.display_manager import DisplayManager
from managers.phase_manager import PhaseManager
from managers.prompt_manager import PromptManager
from managers.llm_manager import LLMManager
from utils.case_importer import import_cases_from_csv
from models.phase import PhaseType
from models.assessment import TopicAssessment, CoverageAssessment, TopicRelevance

st.set_page_config(layout="wide")

# Constants
GOOGLE_SHEET_URL = st.secrets["gcp"]["SHEET_URL"]
MAX_CASES_PER_DAY = 3
EMAIL_ADDRESS = st.secrets["email"]["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["email"]["EMAIL_PASSWORD"]

# Session State Initialization
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "email_verified" not in st.session_state:
    st.session_state.email_verified = False
if "pending_registration" not in st.session_state:
    st.session_state.pending_registration = {}

# Connect to Google Sheet
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
credentials = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=SCOPES
)

gc = gspread.authorize(credentials)
sh = gc.open_by_url(GOOGLE_SHEET_URL)
credentials_ws = sh.worksheet("Credentials")
usage_ws = sh.worksheet("UsageLog")

# Utility Functions
def send_verification_email(to_email, code):
    msg = EmailMessage()
    msg["Subject"] = "Socratic AI Verification Code"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg.set_content(f"Your verification code is: {code}")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

@st.cache_data(ttl=30)
def load_credentials():
    records = credentials_ws.get_all_records()
    creds = {"users": {}}
    for r in records:
        creds["users"][r["username"]] = {
            "email": r["email"],
            "name": r["name"],
            "password": r["password_hash"],
        }
    return creds

def save_new_credential(username, email, name, password_hash):
    credentials_ws.append_row([username, email, name, password_hash])

@st.cache_data(ttl=30)
def load_usage_log():
    records = usage_ws.get_all_records()
    usage_data = {}
    for r in records:
        username = r["username"]
        date = r["date"]
        count = int(r["count"])
        usage_data.setdefault(username, {})[date] = count
    return usage_data

def save_usage(username, date, count):
    usage_ws.append_row([username, date, count])

def update_usage(username, date, count):
    cells = usage_ws.findall(username)
    for cell in cells:
        row = cell.row
        row_data = usage_ws.row_values(row)
        if len(row_data) >= 2 and row_data[1] == date:
            usage_ws.update_cell(row, 3, count)
            return
    save_usage(username, date, count)

# Account Management UI
if not st.session_state.logged_in:
    st.sidebar.title("Account Management")
    creds = load_credentials()

    with st.sidebar.expander("Create New Account"):
        new_username = st.text_input("Username", key="reg_user")
        new_email = st.text_input("Email", key="reg_email")
        new_name = st.text_input("Name", key="reg_name")
        new_password = st.text_input("Password", type="password", key="reg_pw")

        if st.button("Send Verification Code"):
            if not all([new_username, new_email, new_name, new_password]):
                st.warning("All fields are required.")
            elif not is_valid_email(new_email):
                st.warning("Please enter a valid email address.")
            elif new_username in creds["users"]:
                st.warning("Username already exists.")
            elif any(user.get("email") == new_email for user in creds["users"].values()):
                st.warning("Email already associated with an account.")
            else:
                code = str(random.randint(100000, 999999))
                st.session_state.pending_registration = {
                    "username": new_username,
                    "email": new_email,
                    "name": new_name,
                    "password": new_password,
                    "code": code
                }
                send_verification_email(new_email, code)
                st.session_state.email_verified = False
                st.success("Verification code sent to your email. Please check your spam")

        if st.session_state.get("pending_registration"):
            input_code = st.text_input("Enter the verification code sent to your email. Please check your spam")
            if st.button("Verify and Register"):
                reg = st.session_state.pending_registration
                if input_code == reg.get("code"):
                    creds = load_credentials()
                    if reg["username"] in creds["users"] or any(user.get("email") == reg["email"] for user in creds["users"].values()):
                        st.warning("Username or email already associated with an account.")
                    else:
                        save_new_credential(
                            reg["username"],
                            reg["email"],
                            reg["name"],
                            hash_password(reg["password"])
                        )
                        st.success("Account created. Please log in.")
                        st.session_state.pending_registration = {}
                        st.session_state.email_verified = True
                else:
                    st.error("Incorrect verification code.")

    st.title("Clinical Case Tutor")
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        creds = load_credentials()
        user = creds["users"].get(username)
        if not user:
            st.error("User not found.")
        elif check_password(password, user["password"]):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.name = user["name"]
            st.success(f"Welcome {user['name']}!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error("Incorrect password.")

# Logged-In Sidebar
if st.session_state.logged_in:
    with st.sidebar:
        st.success(f"Account: {st.session_state.name}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.name = ""
            st.rerun()

# Main App
if st.session_state.logged_in:
    today = str(datetime.date.today())
    username = st.session_state.username

    usage_data = load_usage_log()
    current_usage = usage_data.get(username, {}).get(today, 0)

    def increment_usage():
        usage_data.setdefault(username, {}).setdefault(today, 0)
        usage_data[username][today] += 1
        st.session_state.current_usage = usage_data[username][today] 
        update_usage(username, today, usage_data[username][today])

    class RAGSearchBar:
        """
        A Retrieval Augmented Generation search bar for clinical case application.
        Uses FAISS indexes for fast similarity search and OpenAI for embedding generation.
        """
        
        def __init__(self, client, embedding_model: str = 'text-embedding-3-large', chat_model: str = 'gpt-4'):

            """
            Initialize the RAG search bar with OpenAI client and embedding model.
            
            Args:
                client: OpenAI client for generating embeddings
                embedding_model: The embedding model to use
            """
            self.client = client
            self.embedding_model = embedding_model
            self.chat_model = chat_model
            self.search_indexes = {
                "PoC": {"name": "Patient Presentation", "weight": 1.0},
                "DDx": {"name": "Differential Diagnosis", "weight": 0.8},
                "PD": {"name": "Physical Exam", "weight": 0.7},
                "TD": {"name": "Test Results", "weight": 0.6},
                "FD": {"name": "Final Diagnosis", "weight": 0.5}
            }
            
            # Load case data
            self.load_case_data()
        
        def load_case_data(self): # Debugging
            """Load case data from CSV if available."""
            try:
                self.cases_df = pd.read_csv("data_files/all_cases.csv")
                print(f"Loaded {len(self.cases_df)} cases from all_cases.csv")
            except:
                print("Warning: all_cases.csv not found. Some functions may be limited.")
                self.cases_df = None
        
        def generate_embeddings(self, texts):
            """Generate embeddings for a list of texts using OpenAI's API."""
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                st.error(f"Error generating embeddings: {str(e)}")
                return []
            
        def normalize_query(self, query: str) -> str:
            """
            Normalize a query to standard clinical terminology using the OpenAI LLM.
            
            Args:
                query: Raw search string from user
            
            Returns:
                A cleaned, standardized version of the query
            """
            try:
                prompt = f"""
                The following is a user input meant to search for a medical case:
                "{query}"
                
                Normalize this into a concise clinical term or standard phrasing.
                Expand acronyms, convert lay terms to medical language (e.g. 'heart attack' to 'myocardial infarction'), and ensure it’s semantically robust for use in clinical retrieval.
                
                Only return the normalized version. Avoid extra explanation.
                """
                
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=50
                )
                normalized = response.choices[0].message.content.strip()
                return normalized
            except Exception as e:
                st.warning(f"Query normalization failed: {str(e)}")
                return query
        
        def multi_index_search(self, query: str, top_k: int = 10):
            """
            Search across multiple indexes and combine results with weighted scoring.
            
            Args:
                query: The search query text
                top_k: Number of top results to return
                
            Returns:
                DataFrame of combined search results with relevance scores
            """
            if not query.strip():
                return pd.DataFrame()
        
            # Normalize the query first
            normalized_query = self.normalize_query(query)
            # st.caption(f"Interpreted as: `{normalized_query}`")
            
            # Generate embedding for the normalized query
            query_embedding = self.generate_embeddings([normalized_query])[0]
            query_vector = np.array(query_embedding).reshape(1, -1).astype('float32')
            
            # Store results from each index
            all_results = []
            
            # Search in each index
            for index_name, index_info in self.search_indexes.items():
                try:
                    # Check if FAISS index and dataframe files exist
                    index_path = f"data_files/{index_name}_faiss.index"
                    df_path = f"data_files/{index_name.lower()}_df_ada3.pkl"
                    
                    if not os.path.exists(index_path) or not os.path.exists(df_path):
                        continue
                    
                    # Load FAISS index and dataframe
                    index = faiss.read_index(index_path)
                    with open(df_path, 'rb') as f:
                        df = pickle.load(f)
                    
                    # Search for similar vectors
                    distances, indices = index.search(query_vector, top_k)
                    
                    # Convert to dataframe and add search context
                    if len(indices[0]) > 0:  # Check if we got any results
                        results = pd.DataFrame({
                            'case': [df['label'].iloc[i] for i in indices[0]],
                            'distance': distances[0],
                            'relevance': 1 / (1 + distances[0]) * index_info["weight"],  # Convert distance to score
                            'source': index_info["name"],
                            'source_text': [df['text'].iloc[i] for i in indices[0]]
                        })
                        
                        all_results.append(results)
                except Exception as e:
                    st.warning(f"Error searching in {index_name} index: {str(e)}")
            
            if not all_results:
                return pd.DataFrame()
                
            # Combine all results
            combined_results = pd.concat(all_results, ignore_index=True)
            
            # Get the case metadata and join with results
            if self.cases_df is not None:
                try:
                    combined_results = pd.merge(
                        combined_results, 
                        self.cases_df[['case', 'title', 'one_line', 'specialties', 'keywords']], 
                        on='case', 
                        how='left'
                    )
                except Exception as e:
                    st.warning(f"Error merging case metadata: {str(e)}")
            
            # Sort by score (descending) and remove duplicates
            combined_results = combined_results.sort_values('relevance', ascending=False)
            combined_results = combined_results.drop_duplicates(subset=['case'])
            
            return combined_results.head(top_k)
        
        def search_and_recommend(self, query: str, top_k: int = 5):
            """
            Search for relevant cases and generate a recommendation summary.
            
            Args:
                query: The search query text
                top_k: Number of top results to return
                
            Returns:
                Tuple of (search results DataFrame, recommendation text)
            """
            # Search for relevant cases
            results = self.multi_index_search(query, top_k=top_k)
            
            if results.empty:
                return results, "No matching cases found. Try a different search query."
            
            # Create a summary of the search results using GPT
            try:
                summary_prompt = f"""
                A healthcare professional searched for: "{query}"
                
                Top {len(results)} matching cases:
                {results[['title', 'one_line', 'source', 'relevance']].to_string(index=False)}
                
                Provide a brief (3-5 sentences) summary of the search results, explaining why these cases might be relevant 
                to the query and what patterns or insights emerge from these results. Focus on clinical relevance.
                """
                
                response = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.2,
                    max_tokens=250
                )
                
                summary = response.choices[0].message.content
            except Exception as e:
                summary = f"Search complete. Found {len(results)} potentially relevant cases."
            
            return results, summary
        
        def render_search_bar(self):
            """Render the search bar UI component in Streamlit."""
            st.markdown("### Clinical Case Search")

            query = st.text_input(
                "Search cases by symptoms, demographics, diagnoses, or findings",
                placeholder="Type your search here",
                key="rag_search_query"
            )

            # Advanced search options in an expander
            with st.expander("Advanced Search Options"):
                top_k = st.slider("Number of results", 3, 20, 7)

                # Index weights
                st.markdown("#### Search Index Weights")
                weights_col1, weights_col2 = st.columns(2)

                with weights_col1:
                    self.search_indexes["PoC"]["weight"] = st.slider(
                        "Patient Presentation", 0.1, 1.0, 1.0, 0.1)
                    self.search_indexes["DDx"]["weight"] = st.slider(
                        "Differential Diagnosis", 0.1, 1.0, 0.8, 0.1)
                    self.search_indexes["PD"]["weight"] = st.slider(
                        "Physical Exam", 0.1, 1.0, 0.7, 0.1)

                with weights_col2:
                    self.search_indexes["TD"]["weight"] = st.slider(
                        "Test Results", 0.1, 1.0, 0.6, 0.1)
                    self.search_indexes["FD"]["weight"] = st.slider(
                        "Final Diagnosis", 0.1, 1.0, 0.5, 0.1)
            
            # Perform search if button is clicked or Enter is pressed
            if query:
                with st.spinner("Searching for relevant cases..."):
                    start_time = time.time()

                    results, summary = self.search_and_recommend(query, top_k)
                    search_time = time.time() - start_time

                    if not results.empty:
                        st.markdown(f"#### Search Results ({len(results)} cases, {search_time:.2f}s)")
                        st.info(summary)

                        # Display results table
                        display_cols = ['title', 'one_line', 'source', 'relevance']
                        if 'specialties' in results.columns:
                            display_cols.append('specialties')

                        results_display = results[display_cols].copy()
                        results_display['relevance'] = results_display['relevance'].round(3)
                        st.dataframe(results_display, use_container_width=True)

                        # Store results in session state
                        st.session_state.last_search_results = results
                    else:
                        st.warning("No matching cases found. Try a different search query.")

    # Add after set_page_config
    st.components.v1.html(
        """
        <script src="https://unpkg.com/react@17/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@17/umd/react-dom.production.min.js"></script>
        <script src="static/js/differential-helper.js"></script>
        """,
        height=0
    )

    @dataclass
    class LogConfig:
        """Configuration for application logging."""
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        level: int = logging.INFO

    def setup_logging(config: LogConfig) -> None:
        """Set up application logging with the specified configuration."""
        logging.basicConfig(
            format=config.format,
            level=config.level
        )

    class ClinicalCaseTutor:
        """Main application class for the Clinical Case Tutor system."""
        
        def __init__(self):
            """Initialize the tutor system and its components."""
            # Set up logging
            setup_logging(LogConfig())
            self.logger = logging.getLogger(__name__)

            azure_config = st.secrets["azure_openai"]
            self.client = AzureOpenAI(
                api_key=azure_config["AZURE_API_KEY"],
                api_version=azure_config["API_VERSION"],
                azure_endpoint=azure_config["AZURE_ENDPOINT"]
            )
            self.chat_deployment = azure_config["AZURE_CHAT_DEPLOYMENT"]
            self.embedding_deployment = azure_config["AZURE_EMBEDDING_DEPLOYMENT"] 
            self.llm_manager = LLMManager(self.client)

            # Initialize managers in correct order
            self.prompt_manager = PromptManager()
            self.display_manager = DisplayManager(skip_page_config=True)
            self.display_manager.phase_transition_handler = self._handle_phase_transition
            self.case_manager = CaseManager()
            
            # Initialize phase manager and differential manager as None
            self.phase_manager = None
            self.differential_manager = None
            
            # Move session state initialization to constructor
            self._ensure_session_state()
            
            # Initialize managers immediately
            self._initialize_managers()
            
            # Only set up case if not showing search page
            if not st.session_state.get('show_search_page', True):
                self._setup_case()
                
            # Initialize additional session state variables for search
            if "last_search" not in st.session_state:
                st.session_state.last_search = ""
            if "applied_filters" not in st.session_state:
                st.session_state.applied_filters = {}
        
        def _ensure_session_state(self):
            """Ensure all required session state variables exist."""
            if "initialized" not in st.session_state:
                st.session_state.initialized = True
                st.session_state.current_case_id = None
                st.session_state.case_loaded = False
                st.session_state.case_data = None
                st.session_state.chat_messages = []
                st.session_state.assessment_cache = {}
                st.session_state.phase_summaries = {}
                st.session_state.differential_diagnosis = []
                st.session_state.current_phase = PhaseType.HISTORY
                st.session_state.differential_manager = None
                st.session_state._session_id = 0
                st.session_state.show_import_dialog = False
                st.session_state.show_search_page = True

        def _initialize_managers(self):
            """Initialize all managers with current session state."""
            if not hasattr(st.session_state, 'case_data') or not st.session_state.case_data:
                return
                
            self.phase_manager = PhaseManager(
                case_data=st.session_state.case_data,
                llm_manager=self.llm_manager,
                prompt_manager=self.prompt_manager
            )
            
            # Only create a new DifferentialManager if one doesn't exist
            if st.session_state.differential_manager is None:
                self.differential_manager = DifferentialManager(self.llm_manager)
                st.session_state.differential_manager = self.differential_manager
            else:
                self.differential_manager = st.session_state.differential_manager

        def _get_available_cases(self) -> list:
            """Get list of available clinical cases."""
            cases_dir = Path("cases")
            return [f.stem for f in cases_dir.glob("*.json")]
        
        def _load_new_case(self, case_id: str):
            """Load a new clinical case and initialize its managers."""
            try:
                self.logger.info(f"Loading case: {case_id}")
                
                # Only proceed if this is actually a new case
                if st.session_state.current_case_id == case_id:
                    return
                    
                # Load case data first
                case_data = self.case_manager.load_case(case_id)
                self.logger.info(f"Case loaded successfully: {case_id}")
                
                # Reset all state in a single block
                st.session_state.chat_messages = []
                st.session_state.assessment_cache = {}
                st.session_state.phase_summaries = {}
                st.session_state.differential_manager = None
                st.session_state.case_data = case_data
                st.session_state.current_case_id = case_id
                st.session_state.current_phase = PhaseType.HISTORY
                st.session_state.case_loaded = True
                st.session_state.case_presented = False
                st.session_state.show_search_page = False
                
                # Clear any existing phase-related flags
                if 'phase_completion_status' in st.session_state:
                    del st.session_state.phase_completion_status
                if 'summary_generated' in st.session_state:
                    del st.session_state.summary_generated
                if 'pending_next_phase' in st.session_state:
                    del st.session_state.pending_next_phase
                    
                # Reinitialize managers with new case data
                self._initialize_managers()
                    
                # Force a complete rerun with new session ID to ensure fresh state
                st.session_state['_session_id'] = st.session_state.get('_session_id', 0) + 1
                
                # Debug log to confirm the case is loaded
                self.logger.info(f"Case {case_id} loaded, rerunning app...")
                
                # Use this for debugging
                st.session_state.debug_message = f"Loaded case: {case_id}"
                
                time.sleep(3)
                st.rerun()
                    
            except Exception as e:
                self.logger.error(f"Error loading case: {str(e)}")
                st.error(f"Error loading case: {str(e)}")

        def _setup_case(self):
            """Set up or continue the current clinical case."""
            # If show_search_page is True, don't do anything here
            if st.session_state.get('show_search_page', True):
                return
                
            available_cases = self._get_available_cases()
            
            # with st.sidebar:
            #     # Search input field
            #     search_query = st.text_input("🔍 Search Cases", key="case_search")
                
            #     if search_query:
            #         # Perform search
            #         search_results = self.case_manager.search_cases(search_query)
            #         if search_results:
            #             # Create a descriptive list of search results
            #             case_options = []
            #             case_id_map = {}
                        
            #             for case in search_results:
            #                 case_id = case["id"]
            #                 display_text = f"{case_id}: {case['title']}"
            #                 case_options.append(display_text)
            #                 case_id_map[display_text] = case_id
                        
            #             # Use selectbox for search results
            #             selected_option = st.selectbox(
            #                 "Search Results", 
            #                 case_options,
            #                 key=f"search_results_{st.session_state.get('_session_id', 0)}"
            #             )
                        
            #             selected_case = case_id_map[selected_option]
                        
            #             # Only load if the selection changed
            #             if selected_case != st.session_state.current_case_id:
            #                 self._load_new_case(selected_case)
            #         else:
            #             st.info("No matching cases found.")
            #             # Show all cases as fallback
            #             self._display_regular_case_selector(available_cases)
            #     else:
            #         # Regular case selector when not searching
            #         self._display_regular_case_selector(available_cases)
            
            # def _display_regular_case_selector(self, available_cases):
            #     """Display the regular case selector dropdown."""
            #     # Use a timestamp-based key to ensure fresh state
            #     selectbox_key = f"case_selector_{st.session_state.get('_session_id', 0)}"
                
            #     current_index = 0
            #     if st.session_state.current_case_id in available_cases:
            #         current_index = available_cases.index(st.session_state.current_case_id)
                    
            #     selected_case = st.selectbox(
            #         "Select Clinical Case",
            #         options=available_cases,
            #         index=current_index,
            #         key=selectbox_key
            #     )
                
            #     # Only trigger case load if selection actually changed
            #     if selected_case != st.session_state.current_case_id:
            #         self._load_new_case(selected_case)

        def _display_regular_case_selector(self, available_cases):
            """Display the regular case selector dropdown."""
            # Use a timestamp-based key to ensure fresh state
            selectbox_key = f"case_selector_{st.session_state.get('_session_id', 0)}"
            
            current_index = 0
            if st.session_state.current_case_id in available_cases:
                current_index = available_cases.index(st.session_state.current_case_id)
                
            selected_case = st.selectbox(
                "Select Clinical Case",
                options=available_cases,
                index=current_index,
                key=selectbox_key
            )
            
            # Only trigger case load if selection actually changed
            if selected_case != st.session_state.current_case_id:
                self._load_new_case(selected_case)

        # def _show_import_dialog(self):
        #     """Show dialog to import cases from CSV."""
        #     st.sidebar.subheader("Import Cases")
            
        #     csv_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            
        #     case_limit = st.sidebar.number_input("Number of cases to import", min_value=1, max_value=100, value=50)
            
        #     if st.sidebar.button("Start Import"):
        #         if csv_file:
        #             # Save the uploaded file temporarily
        #             temp_csv_path = Path("temp_cases.csv")
        #             with open(temp_csv_path, "wb") as f:
        #                 f.write(csv_file.getbuffer())
                    
        #             try:
        #                 # Import the cases
        #                 imported_cases = import_cases_from_csv(str(temp_csv_path), "cases", limit=case_limit)
        #                 st.sidebar.success(f"Successfully imported {len(imported_cases)} cases!")
                        
        #                 # Clean up the temp file
        #                 temp_csv_path.unlink()
                        
        #                 # Reset session to show new cases
        #                 st.session_state._session_id = st.session_state.get('_session_id', 0) + 1
                        
        #                 # Close the dialog
        #                 st.session_state.show_import_dialog = False
                        
        #                 # Force a rerun to refresh the UI
        #                 st.rerun()
                        
        #             except Exception as e:
        #                 st.sidebar.error(f"Error importing cases: {str(e)}")
        #         else:
        #             st.sidebar.error("Please upload a CSV file.")
            
        #     if st.sidebar.button("Cancel"):
        #         st.session_state.show_import_dialog = False
        #         st.rerun()
                
        def _display_initial_prompt(self):
            """Display the opening prompt for the current phase."""
            if not st.session_state.case_loaded or not self.phase_manager:
                return

            try:
                self.logger.info("Displaying initial prompt for the current phase")
                case_data = st.session_state.case_data
                
                # Initialize chat history if needed
                if 'chat_messages' not in st.session_state:
                    st.session_state.chat_messages = []
                
                # Show case presentation first
                if not st.session_state.get("case_presented", False):
                    # Just use the title as it contains the presentation
                    presentation = case_data.metadata.title
                    self.logger.info(f"Showing case presentation: {presentation}")
                    
                    # Add presentation to chat history
                    self.display_manager.update_chat_display(
                        message=presentation,
                        role="assistant"
                    )
                    st.session_state.case_presented = True
                
                    # Show the opening prompt
                    if hasattr(self.phase_manager.current_phase, 'config'):
                        opening_prompt = self.phase_manager.current_phase.config.opening_prompt
                        self.logger.info(f"Showing opening prompt: {opening_prompt}")
                        
                        # Add opening prompt to chat history
                        self.display_manager.update_chat_display(
                            message=opening_prompt,
                            role="assistant"
                        )
                
                # Force a rerun to ensure display updates
                st.rerun()
        
            except Exception as e:
                self.logger.error(f"Error displaying initial prompt: {str(e)}")
                st.error("An error occurred while starting the case. Please try again.")
                    
            except Exception as e:
                self.logger.error(f"Error displaying initial prompt: {str(e)}")
                st.error("An error occurred while starting the case. Please try again.")

        def _display_phase_completion_message(self):
            """Display message that phase criteria are met and show summary button."""
            self.display_manager.update_chat_display(
                "✨ You've covered all the key information for this phase! You can continue the discussion, "
                "or click the button below when you're ready to summarize what we've learned.",
                role="assistant"
            )
            
            # Add button for proceeding to summary
            with self.display_manager.chat_col:
                if st.button("Generate Phase Summary", key="generate_summary"):
                    self._generate_phase_summary()
                    st.rerun()

        def _update_initial_display(self):
            """Initialize all display components with current case state."""
            if st.session_state.case_data:
                self.logger.info("Updating initial display with case data")
                self.display_manager.display_case_header(st.session_state.case_data)
                self._update_displays()
                self._display_initial_prompt()
            else:
                self.logger.error("No case data found in session state")
        
        def _generate_phase_summary(self):
            """Generate and display the phase summary."""
            self.logger.info("Generating phase summary...")
            current_phase = self.phase_manager.current_phase_type
            
            # Generate comprehensive summary using chat history
            summaries = self.phase_manager.generate_phase_summary(
                st.session_state.chat_messages
            )
            
            # Store in session state for sidebar display
            if 'phase_summaries' not in st.session_state:
                st.session_state.phase_summaries = {}
            st.session_state.phase_summaries[current_phase] = summaries["phase_summary"]
            
            # Create a more comprehensive summary for chat display
            enhanced_summary = f"""
        ## Phase Summary: {current_phase.value.capitalize()}

        ### Information Retrieved:
        {summaries["phase_summary"].get("retrieved_info", "No summary available.")}

        ### AI Nudges Provided:
        {"- " + "- ".join(summaries["phase_summary"].get("ai_nudges", ["None"])) if summaries["phase_summary"].get("ai_nudges") else "None"}

        ### Information Missed:
        {"- " + "- ".join(summaries["phase_summary"].get("missed_info", ["None"])) if summaries["phase_summary"].get("missed_info") else "None"}

        ### Assessment of Your Differential Diagnosis:
        {summaries["phase_summary"].get("final_ddx_assessment", "No assessment available.")}

        ### Evolution of Your Differential Diagnosis:
        {summaries["phase_summary"].get("ddx_evolution", "No evolution recorded.")}
        """
            
            # Display the enhanced summary
            self.display_manager.update_chat_display(
                enhanced_summary,
                role="assistant"
            )
            
            # Update the case information display with clinical summary
            self.display_manager.update_case_information(
                st.session_state.phase_summaries,
                current_phase
            )
            
            # Get next phase
            phase_sequence = [
                PhaseType.HISTORY,
                PhaseType.PHYSICAL,
                PhaseType.TESTING,
                PhaseType.MANAGEMENT,
                PhaseType.DISCUSSION
            ]
            
            try:
                current_idx = phase_sequence.index(current_phase)
                next_phase = phase_sequence[current_idx + 1] if current_idx + 1 < len(phase_sequence) else None
            except (ValueError, IndexError):
                self.logger.error(f"Invalid phase transition from {current_phase}")
                next_phase = None
            
            if next_phase:
                st.session_state.pending_next_phase = next_phase
                # Mark that summary has been generated
                st.session_state.summary_generated = True
        def _update_displays(self):
            """Update all display components with current case state."""
            if not st.session_state.case_loaded or not self.phase_manager:
                return

            try:
                self.logger.info("Updating displays with current case state")
                
                current_phase = self.phase_manager.current_phase_type
                completed_phases = [
                    phase_type for phase_type in PhaseType 
                    if phase_type in st.session_state.phase_summaries
                ]
                
                # Update phase progress
                self.display_manager.update_phase_progress(
                    current_phase=current_phase,
                    completed_phases=completed_phases
                )
                
                # Update case information
                if st.session_state.phase_summaries:
                    self.display_manager.update_case_information(
                        st.session_state.phase_summaries,
                        current_phase
                    )
                
                # Only update differential if we haven't already in this render cycle
                if self.differential_manager and not hasattr(st.session_state, '_differential_updated'):
                    self.logger.info("Updating differential display")
                    self.display_manager.update_differential_panel(
                        differential_manager=self.differential_manager
                    )
                    st.session_state._differential_updated = True

            except Exception as e:
                self.logger.error(f"Error updating displays: {str(e)}")
                st.error("An error occurred while updating the display. Please try again.")     
                
        def _generate_next_response(self, coverage_assessment) -> str:
            """Generate the next appropriate response based on coverage assessment."""
            phase_context = self.phase_manager.get_phase_context()
            
            # First, check if we should present a teaching point
            if coverage_assessment.newly_covered_points:
                point = coverage_assessment.newly_covered_points[0]
                return self.prompt_manager.generate_teaching_prompt(
                    point.content,
                    self.phase_manager.current_phase_type
                )
            
            # If critical elements are missing, generate a probe for those
            if coverage_assessment.missing_critical_elements:
                return self.prompt_manager.construct_probe_question(
                    coverage_assessment.missing_critical_elements[0],
                    phase_context
                )
            
            # Otherwise, generate a general follow-up question
            return self.prompt_manager.construct_follow_up_question(phase_context)

        def _handle_phase_transition(self):
            """Handle transition to the next phase with enhanced summaries and differential check."""
            if not hasattr(self, 'phase_manager') or not self.phase_manager:
                self.logger.warning("Phase transition called before phase manager initialization")
                return
                    
            if not hasattr(st.session_state, 'chat_messages'):
                self.logger.warning("Phase transition called before chat history initialization")
                return

            self.logger.info("Starting phase transition...")
            current_phase = self.phase_manager.current_phase_type
            
            # Generate comprehensive summary using chat history
            summaries = self.phase_manager.generate_phase_summary(
                st.session_state.chat_messages
            )
            
            # Store in session state for sidebar display
            if 'phase_summaries' not in st.session_state:
                st.session_state.phase_summaries = {}
            st.session_state.phase_summaries[current_phase] = summaries["phase_summary"]
            
            # Check differential diagnosis if we're transitioning from certain phases
            if current_phase in [PhaseType.HISTORY, PhaseType.PHYSICAL, PhaseType.TESTING]:
                try:
                    # Get the current phase object
                    phase = self.phase_manager.current_phase
                    ideal_differential = getattr(phase, "current_ideal_differential_diagnosis", None)
                    
                    if ideal_differential and self.differential_manager:
                        matches_sufficiently, feedback = self.differential_manager.compare_differentials(ideal_differential)
                        
                        # Display feedback about differential
                        self.display_manager.update_chat_display(
                            feedback,
                            role="assistant"
                        )
                        
                        # If differential doesn't match well enough, pause transition
                        if not matches_sufficiently:
                            self.display_manager.update_chat_display(
                                "Please review and update your differential diagnosis based on the feedback above before proceeding.",
                                role="assistant"
                            )
                            return
                except Exception as e:
                    self.logger.error(f"Error comparing differentials: {str(e)}")
                    # Continue with phase transition even if differential comparison fails
            
            # Get next phase using phase sequence
            phase_sequence = [
                PhaseType.HISTORY,
                PhaseType.PHYSICAL,
                PhaseType.TESTING,
                PhaseType.MANAGEMENT,
                PhaseType.DISCUSSION
            ]
            
            try:
                current_idx = phase_sequence.index(current_phase)
                next_phase = phase_sequence[current_idx + 1] if current_idx + 1 < len(phase_sequence) else None
            except (ValueError, IndexError):
                self.logger.error(f"Invalid phase transition from {current_phase}")
                next_phase = None
            
            # Create a more comprehensive summary for chat display
            enhanced_summary = f"""
        ## Phase Summary: {current_phase.value.capitalize()}

        ### Information Retrieved:
        {summaries["phase_summary"].get("retrieved_info", "No summary available.")}

        ### AI Nudges Provided:
        {"- " + "- ".join(summaries["phase_summary"].get("ai_nudges", ["None"])) if summaries["phase_summary"].get("ai_nudges") else "None"}

        ### Information Missed:
        {"- " + "- ".join(summaries["phase_summary"].get("missed_info", ["None"])) if summaries["phase_summary"].get("missed_info") else "None"}

        ### Assessment of Your Differential Diagnosis:
        {summaries["phase_summary"].get("final_ddx_assessment", "No assessment available.")}

        ### Evolution of Your Differential Diagnosis:
        {summaries["phase_summary"].get("ddx_evolution", "No evolution recorded.")}
        """
            
            # Display the enhanced summary
            self.display_manager.update_chat_display(
                enhanced_summary,
                role="assistant"
            )
            
            # Update the case information display with clinical summary
            self.display_manager.update_case_information(
                st.session_state.phase_summaries,
                current_phase
            )
            
            # If there's a next phase, add the transition prompt
            if next_phase:
                transition_prompt = f"\nClick the button below when you are ready to transition to the {next_phase.value.capitalize()} Phase"
                self.display_manager.update_chat_display(
                    transition_prompt,
                    role="assistant"
                )
                # Store next phase in session state for button handling
                st.session_state.pending_next_phase = next_phase
            
            self.logger.info(f"Phase transition complete. Next phase: {next_phase.value if next_phase else 'None'}")

        def _initialize_new_phase(self, new_phase: PhaseType):
            """Initialize everything needed for a new phase."""
            self.logger.info(f"Initializing new phase: {new_phase.value}")
            
            # Update phase in session state FIRST
            st.session_state.current_phase = new_phase
            
            # Only clear assessment cache if we're actually changing phases
            if not hasattr(st.session_state, 'last_phase') or st.session_state.last_phase != new_phase:
                st.session_state.assessment_cache = {}
                st.session_state.last_phase = new_phase
            
            # Reinitialize phase manager with new phase
            if hasattr(self, 'phase_manager'):
                self.phase_manager.current_phase_type = new_phase
                self.phase_manager._initialize_phase()
            
            # Clear chat messages except for case presentation
            if 'chat_messages' in st.session_state:
                initial_presentation = next(
                    (msg for msg in st.session_state.chat_messages 
                    if msg.get("is_presentation", False)), 
                    None
                )
                st.session_state.chat_messages = [initial_presentation] if initial_presentation else []
            
            # Reset assessment cache for new phase
            st.session_state.assessment_cache = {}
            
            # Get the opening prompt from the newly initialized phase
            if hasattr(self.phase_manager.current_phase, 'config'):
                opening_prompt = self.phase_manager.current_phase.config.opening_prompt
                self.logger.info(f"Showing opening prompt for {new_phase.value}: {opening_prompt}")
                self.display_manager.update_chat_display(
                    message=opening_prompt,
                    role="assistant"
                )
        def _generate_next_prompt(self, coverage_assessment):
            """Generate the next appropriate prompt based on current context."""
            if coverage_assessment.newly_covered_points:
                point = coverage_assessment.newly_covered_points[0]
                self.display_manager.display_teaching_point(
                    point.content,
                    self.phase_manager.current_phase.config.teaching_guidance
                )
                
            phase_context = self.phase_manager.get_phase_context()
            next_prompt = self.prompt_manager.construct_system_prompt(phase_context)
            self.display_manager.update_chat_display(next_prompt)
        
        def _case_progress_bar(self, current_phase: str):
            phases = ["History", "Physical", "Testing", "Management", "Discussion"]
            phase_icons = {"Complete": "✓", "Current": "", "Future": ""}
            
            progress_bar = []
            for phase in phases:
                if phase == current_phase:
                    progress_bar.append(f"<b>{phase} (current)</b>")  # Current phase
                elif phases.index(phase) < phases.index(current_phase):
                    progress_bar.append(f"✓ {phase} (completed)")  # Completed phase
                else:
                    progress_bar.append(f"{phase}")  # Future phase
            
            st.markdown(
                f"<div style='text-align: center; font-size: 18px; margin-top: 20px;'>"
                f"{'  →  '.join(progress_bar)}</div>",
                unsafe_allow_html=True,
            )
        def _add_guidance_level_controls_sidebar(self):
            """Add guidance level controls directly to sidebar with robust error handling."""
            try:
                st.sidebar.subheader("Learning Settings")
                
                # Get current guidance level from session state or default to medium
                current_level = st.session_state.get("guidance_level", "medium")
                
                # Create radio buttons for guidance level selection
                guidance_level = st.sidebar.radio(
                    "Tutor Guidance Level",
                    options=["low", "medium", "high"],
                    index=["low", "medium", "high"].index(current_level),
                    help="""
                    Controls how directly the AI assists you:
                    - **Low**: Minimal guidance, no leading questions (advanced learners)
                    - **Medium**: Subtle hints, balanced guidance (intermediate learners)
                    - **High**: Clear guidance, explicit suggestions (novice learners)
                    """
                )
                
                # Only update if the level has changed
                if guidance_level != current_level:
                    # Store the new guidance level in session state
                    st.session_state.guidance_level = guidance_level
                    
                    # Show a temporary message
                    status_container = st.sidebar.empty()
                    status_container.info(f"Updating guidance level to {guidance_level}...")
                    
                    try:
                        # Import GuidanceLevel enum - use try/except to handle import errors
                        try:
                            from managers.prompt_manager import GuidanceLevel
                            guidance_enum = GuidanceLevel(guidance_level)
                        except (ImportError, ValueError) as e:
                            self.logger.error(f"Error importing GuidanceLevel: {str(e)}")
                            # Fallback - use string value directly
                            guidance_enum = guidance_level
                        
                        # Update the prompt manager if available
                        if hasattr(self, 'prompt_manager') and self.prompt_manager is not None:
                            try:
                                self.prompt_manager.set_guidance_level(guidance_enum)
                                self.logger.info(f"Updated prompt_manager guidance to {guidance_level}")
                            except Exception as e:
                                self.logger.error(f"Error updating prompt_manager: {str(e)}")
                        
                        # Update the LLM manager if available
                        if hasattr(self, 'llm_manager') and self.llm_manager is not None:
                            try:
                                self.llm_manager.set_guidance_level(guidance_level)
                                self.logger.info(f"Updated llm_manager guidance to {guidance_level}")
                            except Exception as e:
                                self.logger.error(f"Error updating llm_manager: {str(e)}")
                        
                        # Update the phase manager if available
                        if hasattr(self, 'phase_manager') and self.phase_manager is not None:
                            try:
                                self.phase_manager.set_guidance_level(guidance_enum)
                                self.logger.info(f"Updated phase_manager guidance to {guidance_level}")
                                
                                # Re-initialize the phase if possible
                                try:
                                    self.phase_manager._initialize_phase()
                                    self.logger.info("Re-initialized phase with new guidance level")
                                except Exception as e:
                                    self.logger.error(f"Error re-initializing phase: {str(e)}")
                            except Exception as e:
                                self.logger.error(f"Error updating phase_manager: {str(e)}")
                        
                        # Update status message
                        status_container.success(f"Guidance level updated to: {guidance_level.capitalize()}")
                        
                        # Don't force rerun - this might be causing the crash
                        # Instead, let the user continue with the new guidance level
                        # The next interaction will use the updated level
                        
                    except Exception as e:
                        # Log detailed error and show user-friendly message
                        self.logger.error(f"Error updating guidance level: {str(e)}", exc_info=True)
                        status_container.error(f"Error updating guidance level. Please try again.")
            
            except Exception as e:
                # Catch-all for any other errors
                self.logger.error(f"Error in guidance controls: {str(e)}", exc_info=True)
                st.sidebar.error("Error displaying guidance controls")

        def run(self):
            if hasattr(st.session_state, '_differential_updated'):
                del st.session_state._differential_updated

            proceed_button_already_rendered = False
            
            # Show search page if no case is selected yet or if explicitly requested
            if (not st.session_state.get('case_loaded', False) or 
                not st.session_state.get('case_data') or 
                st.session_state.get('show_search_page', True)):
                
                self._show_search_page()
                return

            # Add sidebar content only when a case is loaded
            with st.sidebar:
                # Clear out the previous sidebar content
                st.empty()
                
                # Add home button
                if st.button("Home", help="Return to case selection"):
                    # Reset case state
                    st.session_state.show_search_page = True
                    # Don't clear the case data yet - just show the search page
                    time.sleep(1)
                    st.rerun()
                
                # Add guidance level controls after home button
                # This is where the guidance controls should go, after the sidebar is initialized
                self._add_guidance_level_controls_sidebar()
                    
                # You can add other sidebar content here that's specific to the case view
                # For example, you might want to show case metadata
                if st.session_state.case_data and hasattr(st.session_state.case_data, 'metadata'):
                    st.subheader("Case Info")
                    metadata = st.session_state.case_data.metadata
                    st.write(f"**ID:** {metadata.id}")
                    if hasattr(metadata, 'difficulty'):
                        st.write(f"**Difficulty:** {metadata.difficulty}")
                    if hasattr(metadata, 'specialties') and metadata.specialties:
                        st.write(f"**Specialty:** {', '.join(metadata.specialties)}")

                    # Skip Phase button
                    if st.button("End Phase & Generate Summary", key="sidebar_skip_button"):
                        self._handle_phase_transition()
                        st.rerun()

                    # Phase Summaries in side panel
                    if 'phase_summaries' in st.session_state and st.session_state.phase_summaries:
                        st.subheader("Case Summary")
                        for phase in PhaseType:
                            summary = st.session_state.phase_summaries.get(phase)
                            if summary:
                                with st.expander(f"{phase.value.capitalize()} Summary", expanded=False):
                                    st.markdown(f"**Summary of Retrieved Information:** {summary.get('retrieved_info', 'No summary available.')}")

                                    nudges = summary.get("ai_nudges", [])
                                    missed = summary.get("missed_info", [])
                                    ddx_assess = summary.get("final_ddx_assessment", "No assessment available.")
                                    ddx_evolution = summary.get("ddx_evolution", "No evolution recorded.")

                                    st.markdown(f"**AI Nudges Provided:** {' '.join(nudges) if nudges else 'None'}")
                                    st.markdown(f"**Missed Information:** {', '.join(missed) if missed else 'None'}")
                                    st.markdown(f"**Assessment of Final DDx:** {ddx_assess}")
                                    st.markdown(f"**DDx Evolution:** {ddx_evolution}")

        
            # If a case is loaded, continue with the normal flow
            self._case_progress_bar(st.session_state.current_phase.value.capitalize())
            st.text("")
            
            # Ensure managers are initialized
            self._initialize_managers()
            
            # Setup layout and continue with normal flow
            if not self.display_manager.chat_col:
                self.display_manager._setup_layout()
                
            if not st.session_state.chat_messages:
                self._update_initial_display()
                
            self._update_displays()
                
                # Handle chat interface
            with self.display_manager.chat_col:
                chat_container = st.container()
                with chat_container:
                    # Display chat messages
                    for msg in st.session_state.chat_messages:
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])
                    if hasattr(st.session_state, 'pending_next_phase') and not proceed_button_already_rendered:
                        next_phase = st.session_state.pending_next_phase
                        unique_key = f"proceed_to_{next_phase.value}_{st.session_state.get('_session_id', 0)}"
                        if st.button(f"Proceed to {next_phase.value.capitalize()} Phase", key=unique_key, type="primary"):
                            self._initialize_new_phase(next_phase)
                            del st.session_state.pending_next_phase
                            if hasattr(st.session_state, 'summary_generated'):
                                del st.session_state.summary_generated
                            if hasattr(st.session_state, 'phase_completion_status'):
                                del st.session_state.phase_completion_status
                            st.rerun()
                        proceed_button_already_rendered = True

                    # Add appropriate button based on state
                    phase_status = getattr(st.session_state, 'phase_completion_status', {})
                    current_phase = self.phase_manager.current_phase_type.value
                    
                    if phase_status.get(current_phase, False):
                        if not getattr(st.session_state, 'summary_generated', False):
                            if st.button(
                                "Generate Phase Summary",
                                key="generate_summary",
                                type="primary"
                            ):
                                self._generate_phase_summary()
                                st.rerun()
                        elif hasattr(st.session_state, 'pending_next_phase') and not proceed_button_already_rendered:
                            next_phase = st.session_state.pending_next_phase
                            if st.button(f"Proceed to {next_phase.value.capitalize()} Phase", key=f"proceed_to_{next_phase.value}", type="primary"):
                                self._initialize_new_phase(next_phase)
                                del st.session_state.pending_next_phase
                                del st.session_state.summary_generated
                                del st.session_state.phase_completion_status
                                st.rerun()
                            proceed_button_already_rendered = True
        
                # Handle user input at bottom of chat
                user_input = st.chat_input("Enter your response...")
                if user_input:
                    self._handle_user_input(user_input)
                    st.rerun()
        
        def _show_search_page(self):
            st.title("Clinical Case Tutor")

            # Initialize RAG search bar once
            if "rag_search_bar" not in st.session_state:
                st.session_state.rag_search_bar = RAGSearchBar(
                    client=self.client,
                    embedding_model=self.embedding_deployment,
                    chat_model=self.chat_deployment
                )

            st.session_state.rag_search_bar.render_search_bar()

            if "current_usage" not in st.session_state:
                usage_data = load_usage_log()
                today = str(datetime.date.today())
                st.session_state.current_usage = usage_data.get(st.session_state.username, {}).get(today, 0)
            
            st.info(f"Cases started today: {st.session_state.current_usage} / {MAX_CASES_PER_DAY}")

            # Handle direct search result selection
            selected_case_id = getattr(st.session_state.rag_search_bar, "selected_case_id", None)
            if selected_case_id:
                st.session_state.rag_search_bar.selected_case_id = None
                if st.session_state.current_usage >= MAX_CASES_PER_DAY:
                    st.error("You’ve reached your daily case limit of 3. Come back tomorrow!")
                else:
                    increment_usage()
                    self._load_new_case(selected_case_id)
                    st.rerun()

            # Load from dropdown result
            if "last_search_results" in st.session_state and not st.session_state.last_search_results.empty:
                st.subheader("Load Case")
                case_options = st.session_state.last_search_results.drop_duplicates(subset="case")
                case_labels = [f"{row['case']}: {row['title']}" for _, row in case_options.iterrows()]
                selected_idx = st.selectbox("Select a case to load:", options=range(len(case_options)), format_func=lambda i: case_labels[i])

                if st.button("Load Selected Case", type="primary", key="load_case_btn"):
                    case_id = case_options.iloc[selected_idx]['case']
                    if st.session_state.current_usage >= MAX_CASES_PER_DAY:
                        st.error("You’ve reached your daily case limit of 3. Come back tomorrow!")
                    else:
                        increment_usage()
                        self._load_new_case(case_id)
                        time.sleep(0.5)
                        st.rerun()

            # Featured cases tabs
            st.subheader("Featured Cases")
            available_cases = self._get_available_cases()
            filters = {"age_min": 0, "age_max": 100, "sex": "All", "difficulty": "All"}
            filtered_cases = self._get_filtered_cases(available_cases, filters)
            if not filtered_cases:
                # fallback to all available metadata
                filtered_cases = []
                for case_id in available_cases:
                    try:
                        with open(Path("cases") / f"{case_id}.json", 'r') as f:
                            case_data = json.load(f)
                        metadata = case_data.get("metadata", {})
                        filtered_cases.append({"id": case_id, "metadata": metadata})
                    except Exception as e:
                        self.logger.error(f"Failed fallback load of {case_id}: {e}")

            tabs = st.tabs(["Recent", "Popular", "Cases By Specialty"])

            # Recent Cases 
            with tabs[0]:
                st.write("Recently added cases:")
                recent_cases = sorted(
                    available_cases,
                    key=lambda x: os.path.getctime(Path("cases") / f"{x}.json"),
                    reverse=True
                )
                for idx, case_id in enumerate(recent_cases):
                    self._render_case_card(case_id, f"recent_{case_id}_{idx}")

            # Popular Cases 
            with tabs[1]:
                st.write("Popular cases:")
                popular_cases = self._get_popular_cases(available_cases)[:5]
                for popular_idx, case_id in enumerate(popular_cases):
                    self._render_case_card(case_id, f"popular_{case_id}_{popular_idx}")

            # All Cases By Specialty 
            with tabs[2]:

                specialty_cases = {}

                for case in filtered_cases:
                    try:
                        case_id = case['id']
                        metadata = case['metadata']
                        specialties = metadata.get('specialties', [])
                        presentation = metadata.get('original_presentation') or metadata.get('title') or case_id

                        if not specialties:
                            specialties = ['Uncategorized']

                        for spec in specialties:
                            specialty_cases.setdefault(spec.strip().title(), []).append((case_id, presentation))

                    except Exception as e:
                        self.logger.error(f"Error processing case {case}: {str(e)}")

                for spec_idx, (specialty, cases) in enumerate(sorted(specialty_cases.items())):
                    with st.expander(f"{specialty} ({len(cases)} cases)"):
                        for case_idx, (case_id, presentation) in enumerate(cases):
                            cols = st.columns([4, 1])
                            with cols[0]:
                                st.write(presentation[:150] + "..." if len(presentation) > 150 else presentation)
                            with cols[1]:
                                if st.button("Select", key=f"all_{spec_idx}_{case_id}_{case_idx}"):
                                    if st.session_state.current_usage >= MAX_CASES_PER_DAY:
                                        st.error("You’ve reached your daily case limit of 3. Come back tomorrow!")
                                    else:
                                        increment_usage()
                                        self._load_new_case(case_id)
                                        time.sleep(0.5)
                                        st.rerun()

        # Helper functions
        def _render_case_card(self, case_id, key):
            try:
                with open(Path("cases") / f"{case_id}.json", 'r') as f:
                    case_data = json.load(f)
                metadata = case_data.get('metadata', {})
                presentation = metadata.get('original_presentation', metadata.get('title', case_id))

                with st.container(border=True):
                    st.markdown(f"**{presentation[:200]}...**" if len(presentation) > 200 else f"**{presentation}**")
                    if 'specialties' in metadata:
                        st.markdown(f"Specialties: {', '.join(metadata['specialties'])}")

                    if st.button(f"Start Case", key=key):
                        if "current_usage" not in st.session_state:
                            usage_data = load_usage_log()
                            today = str(datetime.date.today())
                            st.session_state.current_usage = usage_data.get(st.session_state.username, {}).get(today, 0)

                        if st.session_state.current_usage >= MAX_CASES_PER_DAY:
                            st.error("You’ve reached your daily case limit of 3. Come back tomorrow!")
                        else:
                            increment_usage()
                            self._load_new_case(case_id)
                            time.sleep(4)
                            st.rerun()
            except Exception as e:
                self.logger.error(f"Error loading case {case_id}: {str(e)}")

        # Stand in until we have user data to establish popular cases
        def _get_popular_cases(self, available_cases):
            try:
                return sorted(available_cases)[:10]
            except Exception as e:
                self.logger.warning(f"Fallback to unsorted popular cases due to error: {e}")
                return available_cases[:5]

        def _get_filtered_cases(self, case_ids, filters):
            """Filter cases based on selected criteria"""
            filtered_cases = []
            
            for case_id in case_ids:
                try:
                    case_path = Path("cases") / f"{case_id}.json"
                    with open(case_path, 'r') as f:
                        case_data = json.load(f)
                    metadata = case_data.get('metadata', {})
                    
                    # Extract age and sex information from presentation
                    presentation = metadata.get("original_presentation", "")
                    extracted_age = None
                    extracted_sex = None
                    
                    # Simple extraction logic (can be improved with regex)
                    if "year-old" in presentation.lower():
                        try:
                            age_part = presentation.split("year-old")[0].strip().split()[-1]
                            if age_part.isdigit():
                                extracted_age = int(age_part)
                        except:
                            extracted_age = None
                        
                        if "male" in presentation.lower() and filters["sex"] != "All":
                            extracted_sex = "Male"
                        elif "female" in presentation.lower() and filters["sex"] != "All":
                            extracted_sex = "Female"
                        else:
                            extracted_sex = "Unknown"
                    
                    # Apply filters
                    passes_filters = True
                    
                    # Age filter
                    if extracted_age is not None:
                        if extracted_age < filters["age_min"] or extracted_age > filters["age_max"]:
                            passes_filters = False
                    
                    # Sex filter
                    if filters["sex"] != "All" and extracted_sex != filters["sex"]:
                        passes_filters = False
                    
                    # Difficulty filter
                    if filters["difficulty"] != "All" and metadata.get("difficulty", "") != filters["difficulty"]:
                        passes_filters = False
                    
                    if passes_filters:
                        filtered_cases.append({"id": case_id, "metadata": metadata})
                        
                except Exception as e:
                    self.logger.error(f"Error filtering case {case_id}: {str(e)}")
            
            return filtered_cases
        
        def _handle_user_input(self, user_input: str):
            """Process and respond to user input using conversational context."""
            self.logger.info(f"Handling user input: {user_input}")
            
            if not st.session_state.case_loaded or not self.phase_manager:
                st.error("Please select a case before continuing.")
                return
            
            # Debug shortcut for phase transition
            if user_input.lower() in ["move forward", "next phase", "skip"]:
                self.logger.info("Debug command detected - forcing phase transition")
                self.display_manager.update_chat_display(
                    "Debug command detected - moving to phase transition...",
                    role="assistant"
                )
                self._handle_phase_transition()
                return
                    
            try:
                # First assess if the topic is appropriate
                topic_assessment = self.phase_manager.assess_topic(user_input)
                self.logger.info(f"Topic assessment result: {topic_assessment}")
                
                if topic_assessment.relevance != TopicRelevance.ON_TOPIC:
                    if topic_assessment.redirect_message:
                        self.display_manager.update_chat_display(
                            topic_assessment.redirect_message,
                            role="assistant"
                        )
                    return
                
                # Add user message to chat history first
                self.display_manager.update_chat_display(
                    message=user_input,
                    role="user"
                )
                
                # Get current differential diagnoses
                current_differential = []
                if self.differential_manager:
                    current_differential = self.differential_manager.get_ranked_differential()
                
                # Get phase context for the message
                context = {
                    "chat_history": [
                        {
                            "role": msg["role"],
                            "content": msg["content"],
                            "timestamp": msg["timestamp"].isoformat() if "timestamp" in msg else None
                        }
                        for msg in st.session_state.chat_messages
                    ],
                    "differential_diagnoses": [
                        {
                            "name": dx.name,
                            "order": idx + 1,
                            "notes": self.differential_manager.hypotheses[dx.name].notes if dx.name in self.differential_manager.hypotheses else ""
                        }
                        for idx, dx in enumerate(current_differential)
                    ],
                    "required_elements": {
                        "covered": [e.content for e in self.phase_manager.current_phase.required_elements if e.elicited],
                        "uncovered": [e.content for e in self.phase_manager.current_phase.required_elements if not e.elicited]
                    },
                    "completion_block_rationale": self.phase_manager.last_completion_block_rationale
                }
                
                # Log current required elements status
                self.logger.info("Required elements status:")
                for element in self.phase_manager.current_phase.required_elements:
                    self.logger.info(f"- {element.content}: {'covered' if element.elicited else 'not covered'}")
                
                # Use the system prompt from session state
                system_prompt = st.session_state.get('current_phase_prompt')
                if not system_prompt:
                    # If somehow missing, reconstruct it
                    phase_context = self.phase_manager.get_phase_context()
                    system_prompt = self.phase_manager.prompt_manager.construct_system_prompt(phase_context)
                    st.session_state.current_phase_prompt = system_prompt
                
                # Get conversational response with full context
                context["user_text"] = user_input  # Add user input separately

                # DEBUGGING - REMOVE
                self.logger.info("Final user message payload passed to LLMManager:")#
                self.logger.info(json.dumps(context, indent=2))#


                response = self.llm_manager.get_conversational_response(
                    system_prompt=system_prompt,
                    user_message=json.dumps(context),
                    message_history=st.session_state.chat_messages,
                    temperature=0.7
                )
                
                self.logger.info(f"LLM response received: {response is not None}")
                
                if response:
                    # Add bot response to chat history
                    self.display_manager.update_chat_display(
                        message=response,
                        role="assistant"
                    )
                    
                    # Assess coverage and check completion
                    coverage_assessment = self.phase_manager.assess_coverage(user_input, response)
                    
                    # Update phase completion status if needed
                    phase_status = getattr(st.session_state, 'phase_completion_status', {})
                    current_phase = self.phase_manager.current_phase_type.value
                    
                    if not phase_status.get(current_phase, False):
                        is_complete = self.phase_manager.check_phase_completion(st.session_state.chat_messages)
                        if is_complete:
                            self.logger.info("Phase complete! Displaying completion message...")
                            self._display_phase_completion_message()
                            return
                else:
                    self.logger.error("No response received from LLM")
                    st.error("I apologize, but I wasn't able to generate a response. Please try again.")
                    return
                    
                self._update_displays()
            
            except Exception as e:
                self.logger.error(f"Error handling user input: {str(e)}", exc_info=True)
                st.error("An error occurred while processing your input. Please try again.")

    if __name__ == "__main__":
        tutor = ClinicalCaseTutor()
        tutor.run()

