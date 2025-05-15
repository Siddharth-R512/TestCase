import streamlit as st
import re
import logging
import pandas as pd
import io
import time
import hashlib
import fitz  # PyMuPDF
from docx import Document
import json
import os
from typing import List, Dict, Any, Optional, Callable, Union

# Import our custom modules
from azure_llm_handler import AzureFoundryHandler
from prompt_templates import (
    create_test_case_prompt, 
    get_focus_areas_for_chunk, 
    extract_user_story_components,
    get_optimal_temperature
)
from test_case_utils import (
    parse_test_case_output, 
    format_test_case_fields, 
    filter_test_cases_by_priority,
    create_test_case_dataframe
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Test Case Generator",
    page_icon="🧪",
    layout="wide"
)
# Hide the Streamlit Main Menu and Footer
hide_streamlit_style = """
    <style>
        .stAppDeployButton {
                visibility: hidden;
            }
        /* Hide the hamburger menu (main menu) */
        #MainMenu {visibility: hidden;}
        
        /* Hide the footer */
        footer {visibility: hidden;}
    </style>
"""

# Inject the CSS into the Streamlit app's UI
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize session state variables
if 'user_story_test_cases' not in st.session_state:
    st.session_state.user_story_test_cases = {}
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'current_story_id' not in st.session_state:
    st.session_state.current_story_id = None
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.2
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 4096
if 'priority_filter' not in st.session_state:
    st.session_state.priority_filter = "All"
# New session state variables
if 'generation_mode' not in st.session_state:
    st.session_state.generation_mode = "Specific Number"
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'stop_generation' not in st.session_state:
    st.session_state.stop_generation = False
if 'user_story_text' not in st.session_state:
    st.session_state.user_story_text = ""
if 'has_generated' not in st.session_state:
    st.session_state.has_generated = False


# Get LLM handler
@st.cache_resource
def get_llm_handler(temperature=None, max_tokens=None):
    """
    Create or retrieve a cached Azure AI Foundry handler instance
    
    Args:
        temperature: Optional temperature to override session state
        max_tokens: Optional max_tokens to override session state
        
    Returns:
        AzureFoundryHandler instance
    """
    # Try to get Azure credentials from environment variables first
    api_key = os.environ.get("AZURE_AI_FOUNDRY_API_KEY", None)
    endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT", None)
    deployment_name = os.environ.get("AZURE_AI_FOUNDRY_DEPLOYMENT_NAME", "phi-3.5-vision-instruct")
    
    # If not found in environment, try Streamlit secrets
    if not api_key or not endpoint:
        try:
            # Only attempt to access secrets if running in Streamlit context
            api_key = api_key or st.secrets.get("AZURE_AI_FOUNDRY_API_KEY", None)
            endpoint = endpoint or st.secrets.get("AZURE_AI_FOUNDRY_ENDPOINT", None)
            deployment_name = deployment_name or st.secrets.get("AZURE_AI_FOUNDRY_DEPLOYMENT_NAME", "phi-3.5-vision-instruct")
        except Exception as e:
            # Handle case where secrets aren't available (e.g. local dev without .streamlit/secrets.toml)
            logger.warning(f"Could not access Streamlit secrets: {str(e)}")
    
    # If still not found, show a detailed error
    if not api_key or not endpoint:
        st.error("Azure AI Foundry credentials not found. Please configure them as described below.")
        st.info("""
        ### How to configure your credentials:
        
        **For local development:**
        1. Set environment variables before running Streamlit:
           ```
           export AZURE_AI_FOUNDRY_API_KEY=your_api_key
           export AZURE_AI_FOUNDRY_ENDPOINT=your_endpoint_url
           ```
           
        2. Or create a `.streamlit/secrets.toml` file in your project root:
           ```
           AZURE_AI_FOUNDRY_API_KEY = "your_api_key"
           AZURE_AI_FOUNDRY_ENDPOINT = "your_endpoint_url"
           ```
           
        **For Azure deployment:**
        The credentials should be set as Application Settings in your Azure App Service.
        """)
        st.stop()#
    
    handler = AzureFoundryHandler(
        api_key=api_key,
        endpoint=endpoint,
        deployment_name=deployment_name,
        temperature=temperature if temperature is not None else st.session_state.temperature,
        max_tokens=max_tokens if max_tokens is not None else st.session_state.max_tokens
    )
    
    if not handler.check_connection():
        st.error("Failed to connect to Azure AI Foundry. Please check your credentials and endpoint URL.")
        st.info("Make sure you've set valid API key and endpoint values.")
        st.stop()
    
    return handler

# File handling functions
def extract_text_from_pdf(file):
    """Extract text content from a PDF file"""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    """Extract text content from a DOCX file"""
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file):
    """Extract text content from a TXT file"""
    return file.read().decode("utf-8")

# UI update functions
def update_table(test_cases, requested_count, table_placeholder, status_placeholder):
    """
    Update the table display with current test cases
    
    Args:
        test_cases: List of test case dictionaries
        requested_count: Total number of test cases requested
        table_placeholder: Streamlit placeholder for the table
        status_placeholder: Streamlit placeholder for status messages
    """
    # Create DataFrame using utility function
    df = create_test_case_dataframe(test_cases)
    
    # Apply priority standardization if needed
    if "Priority" in df.columns:
        # Get the current priority filter from session state
        current_priority = st.session_state.priority_filter
        
        if current_priority != "All":
            # Force all priorities to match the filter
            df["Priority"] = current_priority
    
    # Display the table
    table_placeholder.table(df)
    
    # Check if we have fewer test cases than requested
    if len(test_cases) > 0 and st.session_state.generation_mode == "Specific Number" and len(test_cases) < requested_count:
        status_placeholder.warning(f"Generated {len(test_cases)} test cases so far. Processing to reach {requested_count}...")
    elif len(test_cases) >= requested_count or st.session_state.generation_mode == "Comprehensive Coverage":
        status_placeholder.success(f"✅ Generated {len(test_cases)} test cases successfully!")

# Function to show the download, regenerate, and reset buttons
def show_result_buttons(test_cases):
    """
    Show the Download, Regenerate, and Reset buttons

    Args:
        test_cases: The generated test cases
    """
    col1, col2 = st.columns([1, 1])

    with col1:
        # Download button
        excel_data = export_to_excel(test_cases)
        if excel_data:
            st.download_button(
                label="Download as Excel",
                data=excel_data,
                file_name="user_story_test_cases.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with col2:
        # Reset button
        if st.button("Reset", use_container_width=True):
            # # Clear session state
            # for key in list(st.session_state.keys()):
            #     del st.session_state[key]
            
            # # Set the reset flag
            # st.session_state.reset_app = True
            
            # Inject meta refresh tag to reload the page immediately
            st.markdown("""
            <meta http-equiv="refresh" content="0">
            """, unsafe_allow_html=True)


# Generate comprehensive test cases
def generate_comprehensive_test_cases(user_story, max_test_count, actor, action, purpose, req_text, crit_text, priority_filter="All"):
    """
    Generate as many high-quality test cases as possible for comprehensive coverage
    
    Args:
        user_story: User story text
        max_test_count: Maximum number of test cases to generate (for safety)
        actor, action, purpose: User story components
        req_text, crit_text: Requirements and acceptance criteria
        priority_filter: Priority filter to apply
        
    Returns:
        List of generated test cases
    """
    # Set up UI elements
    status = st.empty()
    table_placeholder = st.empty()
    
    status.info("Starting comprehensive test case generation. This will generate as many test cases as possible...")
    
    # Initialize
    all_test_cases = []
    batch_size = 10 if priority_filter != "All" else 20  # Generate in batches of 20
    st.session_state.is_generating = True
    st.session_state.stop_generation = False
    
    # Create a special prompt that encourages comprehensive coverage
    prompt = create_test_case_prompt(
        user_story,
        batch_size,
        actor,
        action,
        purpose,
        req_text,
        crit_text,
        start_id=1,
        focus_area="Generate comprehensive test cases covering all aspects of the user story.",
        priority_filter=priority_filter
    )
    
    # Store the prompt in session state
    st.session_state.current_prompt = prompt
    
    # Initialize for first batch
    raw_output = ""
    
    def process_chunk(chunk):
        nonlocal raw_output
        raw_output += chunk
        test_cases = parse_test_case_output(raw_output)
        
        if priority_filter != "All":
            test_cases = filter_test_cases_by_priority(test_cases, priority_filter)
        
        update_table(all_test_cases + test_cases, batch_size, table_placeholder, status)
        
        # Check if should stop
        if st.session_state.stop_generation:
            return
    
    # First batch with relatively low temperature for foundational cases
    llm = get_llm_handler(temperature=0.2, max_tokens=4096)
    llm.generate_test_cases(prompt, callback=process_chunk)
    
    # Get the first batch of test cases
    first_batch = parse_test_case_output(raw_output)
    
    if priority_filter != "All":
        first_batch = filter_test_cases_by_priority(first_batch, priority_filter)
    
    # Process test cases
    for tc in first_batch:
        format_test_case_fields(tc)
    
    all_test_cases.extend(first_batch)
    
    # Update the table
    update_table(all_test_cases, batch_size, table_placeholder, status)
    
    # Continue generating batches until stopped or max reached
    batch_num = 2
    while len(all_test_cases) < max_test_count and not st.session_state.stop_generation:
        status.info(f"Generating batch {batch_num} of test cases for comprehensive coverage...")
        
        # Create a new prompt for this batch with focus on different areas
        focus_area = get_focus_areas_for_chunk(len(all_test_cases), max_test_count)
        
        next_batch_prompt = create_test_case_prompt(
            user_story,
            batch_size,
            actor,
            action,
            purpose,
            req_text,
            crit_text,
            start_id=len(all_test_cases) + 1,
            focus_area=f"Focus on generating NEW test cases different from previous ones. {focus_area}",
            priority_filter=priority_filter
        )
        
        # Store the updated prompt
        st.session_state.current_prompt = next_batch_prompt
        
        # Increase temperature with each batch for more creativity
        temperature = min(0.2 + (batch_num * 0.1), 0.9)
        
        # Generate next batch
        raw_output = ""
        llm = get_llm_handler(temperature=temperature, max_tokens=4096)
        llm.generate_test_cases(next_batch_prompt, callback=process_chunk)
        
        # Parse and filter
        next_batch = parse_test_case_output(raw_output)
        
        if priority_filter != "All":
            next_batch = filter_test_cases_by_priority(next_batch, priority_filter)
        
        # Process test cases
        for tc in next_batch:
            format_test_case_fields(tc)
        
        # Add only unique test cases (avoid duplicates)
        existing_ids = set(tc["Test ID"] for tc in all_test_cases)
        existing_descriptions = set(tc["Description"].lower() for tc in all_test_cases)
        
        unique_test_cases = []
        for tc in next_batch:
            # Update the Test ID to ensure continuity
            tc["Test ID"] = f"TC_{len(all_test_cases) + len(unique_test_cases) + 1:03d}"
            
            # Check if description is unique (case-insensitive comparison)
            if tc["Description"].lower() not in existing_descriptions:
                unique_test_cases.append(tc)
                existing_descriptions.add(tc["Description"].lower())
        
        # If we've stopped getting unique test cases, we may have exhausted possibilities
        if not unique_test_cases:
            status.info("No more unique test cases generated. Comprehensive coverage is complete.")
            break
        
        all_test_cases.extend(unique_test_cases)
        
        # Update the table
        update_table(all_test_cases, len(all_test_cases), table_placeholder, status)
        
        batch_num += 1
        
        # Check if we've reached a reasonable number of test cases
        if len(all_test_cases) >= 50 and batch_num >= 4:
            status.info(f"Generated {len(all_test_cases)} test cases. Continuing to find more unique test cases...")
    
    # Final update
    if st.session_state.stop_generation:
        status.warning(f"Test case generation stopped by user. Generated {len(all_test_cases)} test cases.")
    else:
        status.success(f"Comprehensive test case generation complete! Generated {len(all_test_cases)} test cases.")
    
    # Update session state
    st.session_state.is_generating = False
    st.session_state.just_stopped = True
    st.session_state.generation_complete = True
    st.session_state.user_story_test_cases[st.session_state.current_story_id] = {
        "raw_output": "Comprehensive generation",
        "test_cases": all_test_cases
    }
    
    return all_test_cases

# Updated single batch function with stop button
def generate_test_cases_single_batch(user_story, test_count, actor, action, purpose, req_text, crit_text, priority_filter="All"):
    """
    Generate test cases in a single batch with stop button functionality
    
    Args:
        user_story: User story text
        test_count: Number of test cases to generate
        actor, action, purpose: User story components
        req_text, crit_text: Requirements and acceptance criteria
        priority_filter: Priority filter to apply
        
    Returns:
        List of generated test cases
    """
    # Set up the UI elements
    status = st.empty()
    table_placeholder = st.empty()
    
    status.info(f"Starting test case generation with priority: {priority_filter}...")
    
    # Set generation flags
    st.session_state.is_generating = True
    st.session_state.stop_generation = False
    
    # Create the prompt
    prompt = create_test_case_prompt(user_story, test_count, actor, action, purpose, req_text, crit_text, priority_filter=priority_filter)
    
    # Store the prompt in session state
    st.session_state.current_prompt = prompt
    
    # Initialize storage for parsed test cases
    test_cases = []
    raw_output = ""
    
    # Get optimal temperature
    temperature = get_optimal_temperature(test_count)
    
    # Create callback function for processing chunks
    def process_chunk(chunk):
        nonlocal raw_output, test_cases
        
        # Add to raw output
        raw_output += chunk
        
        # Process the entire raw output each time
        test_cases = parse_test_case_output(raw_output)

        # Apply priority filter if needed
        if priority_filter != "All":
            original_count = len(test_cases)
            test_cases = filter_test_cases_by_priority(test_cases, priority_filter)
        
        # Update table with current results
        update_table(test_cases, test_count, table_placeholder, status)
        
        # Check if should stop
        if st.session_state.stop_generation:
            return  # This will stop the callback processing
    
    # Generate test cases with adjusted temperature
    llm = get_llm_handler(temperature=temperature, max_tokens=max(2048, min(4096, test_count * 200)))
    output = llm.generate_test_cases(prompt, callback=process_chunk)
    
    # Final processing
    test_cases = parse_test_case_output(raw_output)
    
    # Apply priority filter if needed
    if priority_filter != "All":
        test_cases = filter_test_cases_by_priority(test_cases, priority_filter)
    
    # Post-process test cases
    for tc in test_cases:
        format_test_case_fields(tc)
    
    # Final update
    if st.session_state.stop_generation:
        status.warning(f"Test case generation stopped by user. Generated {len(test_cases)} of {test_count} requested test cases.")
    else:
        update_table(test_cases, test_count, table_placeholder, status)
    
    # Update session state
    st.session_state.is_generating = False
    st.session_state.just_stopped = True
    st.session_state.user_story_test_cases[st.session_state.current_story_id] = {
        "raw_output": output,
        "test_cases": test_cases
    }
    
    if len(test_cases) >= test_count:
        st.session_state.generation_complete = True
    
    return test_cases

# Updated function to generate test cases in chunks with stop button
def generate_test_cases_in_chunks(user_story, total_test_count, actor, action, purpose, req_text, crit_text, priority_filter="All"):
    """
    Generate a large number of test cases by splitting into manageable chunks with stop button
    
    Args:
        user_story: User story text
        total_test_count: Total number of test cases to generate
        actor, action, purpose: User story components
        req_text, crit_text: Requirements and acceptance criteria
        priority_filter: Priority filter to apply
        
    Returns:
        List of generated test cases
    """
    all_test_cases = []
    remaining = total_test_count
    # Use smaller chunks for better reliability
    chunk_size = min(20, total_test_count)  # Reduced from 30 to 20 for better reliability
    
    # Set up the UI elements
    status = st.empty()
    table_placeholder = st.empty()
    
    status.info(f"Generating {total_test_count} test cases with priority '{priority_filter}' in batches of {chunk_size}...")
    
    # Set generation flags
    st.session_state.is_generating = True
    st.session_state.stop_generation = False
    
    # Start chunking process
    chunk_number = 1
    start_id = 1
    max_attempts = 3  # Maximum attempts per chunk
    max_chunks = 15   # Increased from 10 to 15 to allow more chunks
    
    while remaining > 0 and chunk_number <= max_chunks and not st.session_state.stop_generation:
        current_chunk = min(chunk_size, remaining)
        
        # Get focus area based on progress
        focus_area = get_focus_areas_for_chunk(start_id - 1, total_test_count)
        
        # Create prompt for this chunk with stronger emphasis on exact count
        chunk_prompt = create_test_case_prompt(
            user_story, 
            current_chunk, 
            actor, 
            action, 
            purpose, 
            req_text, 
            crit_text,
            start_id=start_id,
            focus_area=focus_area,
            priority_filter=priority_filter
        )
        
        # Store the current prompt
        st.session_state.current_prompt = chunk_prompt
        
        attempts = 0
        chunk_success = False
        
        while attempts < max_attempts and not chunk_success and not st.session_state.stop_generation:
            # Initialize for this attempt
            raw_output = ""
            chunk_test_cases = []
            
            # Get optimal temperature, increasing with each attempt for more creativity
            temperature = get_optimal_temperature(current_chunk, chunk_number, attempts)
            
            # Define chunk callback
            def process_chunk_output(chunk):
                nonlocal raw_output, chunk_test_cases
                
                # Add to raw output
                raw_output += chunk
                
                # Process the accumulated output
                chunk_test_cases = parse_test_case_output(raw_output)

                if priority_filter != "All":
                    original_count = len(chunk_test_cases)
                    chunk_test_cases = filter_test_cases_by_priority(chunk_test_cases, priority_filter)
                    
                    # If filtering removed many test cases, log it
                    if len(chunk_test_cases) < original_count * 0.7:
                        status.info(f"Filtered out {original_count - len(chunk_test_cases)} test cases that didn't match '{priority_filter}' priority.")
                
                # Update the UI with combined results
                combined_cases = all_test_cases + chunk_test_cases
                update_table(combined_cases, total_test_count, table_placeholder, status)
                
                # Check if should stop
                if st.session_state.stop_generation:
                    return
            
            # Generate this chunk with adjusted settings
            attempt_text = f" (attempt {attempts+1}/{max_attempts})" if attempts > 0 else ""
            status.info(f"Generating batch {chunk_number}: test cases {start_id} to {start_id + current_chunk - 1} with priority '{priority_filter}'{attempt_text}...")
            
            # Increase max_tokens for larger chunks to ensure we get complete output
            max_tokens = min(4096, max(current_chunk * 300, 3000))
            
            llm = get_llm_handler(temperature=temperature, max_tokens=max_tokens)
            llm.generate_test_cases(chunk_prompt, callback=process_chunk_output)
            
            # Check if generation was stopped
            if st.session_state.stop_generation:
                # Process what we have and exit
                if chunk_test_cases:
                    for tc in chunk_test_cases:
                        format_test_case_fields(tc)
                    all_test_cases.extend(chunk_test_cases)
                break
            
            # Process the final chunk output
            chunk_test_cases = parse_test_case_output(raw_output)
            
            if priority_filter != "All":
                chunk_test_cases = filter_test_cases_by_priority(chunk_test_cases, priority_filter)
            
            # Check if we got a reasonable number of test cases (at least 70% of requested)
            if len(chunk_test_cases) >= 0.7 * current_chunk:
                chunk_success = True
            else:
                attempts += 1
                if attempts < max_attempts:
                    status.warning(f"Only generated {len(chunk_test_cases)} of {current_chunk} requested test cases. Retrying with higher creativity...")
                else:
                    status.warning(f"After {max_attempts} attempts, could only generate {len(chunk_test_cases)} test cases for this batch.")
                    # Accept what we have and move on
                    chunk_success = True
            
            # Post-process the chunk test cases
            for tc in chunk_test_cases:
                format_test_case_fields(tc)
            
            # If we got test cases, add them to our collection
            if chunk_test_cases:
                all_test_cases.extend(chunk_test_cases)
                
                # Update the count of test cases obtained
                obtained = len(chunk_test_cases)
                
                # Update remaining and start_id for next chunk
                remaining -= obtained
                start_id += obtained
        
        # If we stopped generation, break the loop
        if st.session_state.stop_generation:
            break
        
        # If we failed to generate any test cases after multiple attempts
        if not chunk_test_cases and attempts >= max_attempts:
            # Force a very small chunk size and try again
            mini_chunk_size = 5
            status.warning(f"Struggling to generate test cases. Trying with a smaller batch of {mini_chunk_size}...")
            
            mini_prompt = create_test_case_prompt(
                user_story, 
                mini_chunk_size,
                actor, 
                action, 
                purpose, 
                req_text, 
                crit_text,
                start_id=start_id,
                focus_area="Focus on generating just a few core test cases for the main functionality.",
                priority_filter=priority_filter
            )
            
            # Store the updated prompt
            st.session_state.current_prompt = mini_prompt
            
            # Try with very high temperature
            llm = get_llm_handler(temperature=0.95, max_tokens=4096)
            raw_output = ""
            
            def mini_chunk_callback(chunk):
                nonlocal raw_output
                raw_output += chunk
                mini_cases = parse_test_case_output(raw_output)
                combined_cases = all_test_cases + mini_cases
                update_table(combined_cases, total_test_count, table_placeholder, status)
                
                # Check if should stop
                if st.session_state.stop_generation:
                    return
            
            llm.generate_test_cases(mini_prompt, callback=mini_chunk_callback)
            
            # Check if generation was stopped
            if st.session_state.stop_generation:
                break
                
            mini_cases = parse_test_case_output(raw_output)
            
            if priority_filter != "All":
                mini_cases = filter_test_cases_by_priority(mini_cases, priority_filter)
            
            for tc in mini_cases:
                format_test_case_fields(tc)
                
            if mini_cases:
                all_test_cases.extend(mini_cases)
                obtained = len(mini_cases)
                remaining -= obtained
                start_id += obtained
            else:
                status.error(f"Failed to generate more test cases. Please try adjusting your user story or temperature settings.")
                break
        
        chunk_number += 1
        
        # If we're very close to the target, generate the final few in one go
        if 0 < remaining <= 10 and not st.session_state.stop_generation:
            status.info(f"Generating final {remaining} test cases to reach the target...")
            
            final_prompt = create_test_case_prompt(
                user_story, 
                remaining,
                actor, 
                action, 
                purpose, 
                req_text, 
                crit_text,
                start_id=start_id,
                focus_area="These are the final test cases to complete the set. Focus on any missing coverage areas.",
                priority_filter=priority_filter
            )
            
            # Update the prompt
            st.session_state.current_prompt = final_prompt
            
            # Use high temperature for creativity
            llm = get_llm_handler(temperature=0.9, max_tokens=4096)
            raw_output = ""
            
            def final_callback(chunk):
                nonlocal raw_output
                raw_output += chunk
                final_cases = parse_test_case_output(raw_output)
                combined_cases = all_test_cases + final_cases
                update_table(combined_cases, total_test_count, table_placeholder, status)
                
                # Check if should stop
                if st.session_state.stop_generation:
                    return
            
            llm.generate_test_cases(final_prompt, callback=final_callback)
            
            # Check if generation was stopped
            if st.session_state.stop_generation:
                break
                
            final_cases = parse_test_case_output(raw_output)
            
            if priority_filter != "All":
                final_cases = filter_test_cases_by_priority(final_cases, priority_filter)
            
            for tc in final_cases:
                format_test_case_fields(tc)
                
            if final_cases:
                all_test_cases.extend(final_cases)
                remaining -= len(final_cases)
                
            break  # Exit the loop as we've tried to generate the final cases
    
    # Final update
    if st.session_state.stop_generation:
        status.warning(f"Test case generation stopped by user. Generated {len(all_test_cases)} of {total_test_count} requested test cases.")
    else:
        update_table(all_test_cases, total_test_count, table_placeholder, status)
        if len(all_test_cases) >= total_test_count:
            status.success(f"✅ Successfully generated all {total_test_count} test cases!")
    
    # Update session state
    st.session_state.is_generating = False
    st.session_state.just_stopped = True
    st.session_state.user_story_test_cases[st.session_state.current_story_id] = {
        "raw_output": "Generated in chunks",
        "test_cases": all_test_cases
    }
    
    if len(all_test_cases) >= total_test_count:
        st.session_state.generation_complete = True
    
    return all_test_cases

# Combined function for test case generation and processing
def generate_and_process_test_cases(user_story, test_count=15, priority_filter="All"):
    """
    Main function to generate and process test cases from a user story
    
    Args:
        user_story: User story text
        test_count: Number of test cases to generate (used for "Specific Number" mode)
        priority_filter: Priority filter to apply ("All", "High", "Medium", "Low")
        
    Returns:
        List of generated test cases
    """
    # Store the user story text for regeneration purposes
    st.session_state.user_story_text = user_story
    
    # Create a unique ID for this story
    story_id = f"Story_{hashlib.md5(user_story.encode()).hexdigest()[:8]}"
    st.session_state.current_story_id = story_id
    st.session_state.priority_filter = priority_filter
    
    # Extract components from user story using the utility function
    actor, action, purpose, req_text, crit_text = extract_user_story_components(user_story)
    
    # Handle different generation modes
    if st.session_state.generation_mode == "Comprehensive Coverage":
        # For comprehensive coverage, don't use a fixed count
        # Set a maximum safety limit to prevent infinite generation
        max_test_count = 200
        return generate_comprehensive_test_cases(
            user_story, 
            max_test_count, 
            actor, 
            action, 
            purpose, 
            req_text, 
            crit_text, 
            priority_filter=priority_filter
        )
    else:
        # For specific number, use the given test_count
        use_chunking = test_count > 50
        
        if use_chunking:
            return generate_test_cases_in_chunks(
                user_story, 
                test_count, 
                actor, 
                action, 
                purpose, 
                req_text, 
                crit_text, 
                priority_filter=priority_filter
            )
        else:
            return generate_test_cases_single_batch(
                user_story, 
                test_count, 
                actor, 
                action, 
                purpose, 
                req_text, 
                crit_text, 
                priority_filter=priority_filter
            )

# Export test cases to Excel
def export_to_excel(test_cases):
    """
    Export test cases to Excel format
    
    Args:
        test_cases: List of test case dictionaries
        
    Returns:
        Excel file as bytes
    """
    try:
        # Create DataFrame
        df = pd.DataFrame(test_cases)
        
        # Create Excel file
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name="Test_Cases", index=False)
            
            workbook = writer.book
            worksheet = writer.sheets["Test_Cases"]
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D8E4BC',
                'border': 1
            })
            
            # Apply formatting to header row
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths
            worksheet.set_column('A:A', 10)  # Test ID
            worksheet.set_column('B:B', 25)  # Feature
            worksheet.set_column('C:C', 30)  # Scenario
            worksheet.set_column('D:D', 40)  # Description
            worksheet.set_column('E:E', 40)  # Precondition
            worksheet.set_column('F:F', 50)  # Test Steps
            worksheet.set_column('G:G', 50)  # Expected Results
            worksheet.set_column('H:H', 10)  # Priority
        
        excel_buffer.seek(0)
        return excel_buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error creating Excel file: {str(e)}")
        return None

# Display test cases in expandable format
def display_detailed_test_cases(test_cases):
    """
    Display test cases in an expandable format for better readability
    
    Args:
        test_cases: List of test case dictionaries
    """
    for i, tc in enumerate(test_cases):
        with st.expander(f"{tc['Test ID']}: {tc['Description']}", expanded=(i==0)):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Feature")
                st.write(tc['Feature'])
                
                st.markdown("### Scenario")
                st.write(tc['Scenario'])
                
                st.markdown("### Preconditions")
                st.write(tc['Precondition'])
                
                st.markdown("### Priority")
                # Standardize priority display and add color indicators
                priority = tc['Priority'].strip().lower()
                if "high" in priority:
                    st.markdown("🔴 **High**")
                elif "medium" in priority or "med" in priority:
                    st.markdown("🟠 **Medium**")
                elif "low" in priority:
                    st.markdown("🟢 **Low**")
                else:
                    st.markdown(f"⚪ **{tc['Priority']}**")
            
            with col2:
                st.markdown("### Test Steps")
                st.write(tc['Test Steps'])
                
                st.markdown("### Expected Results")
                st.write(tc['Expected Results'])

# App title and description
st.title("Test Case Generator")
st.markdown("""
Generate comprehensive test cases from user stories and requirements.
Enter your user story below and get detailed test cases in minutes.
""")

# User Story input section
st.header("User Story Input")

st.info("""
Enter a user story in any format. The system will generate test cases based on your input.
You can provide structured or unstructured information about your feature requirements.
""")

# Use st.popover to create a popover element
with st.popover("See User Story Example"):
    st.markdown("""
    ### Example User Story:

    **Title**: User Registration

    **As a** new user **I want to** register an account **so that** I can have a personalized shopping experience and manage my orders.

    **Functional Perspective**:
    1. Input fields: name, email address, password
    2. Validation: required fields, valid email format, password strength
    3. Database: store user information securely
    4. Email: send a confirmation email with a verification link

    **Acceptance Criteria**:
    1. The registration page should allow users to enter their name, email address, and password
    2. The system should validate the input fields and display error messages for invalid entries
    3. The system should send a confirmation email to the provided email address
    4. The user should be able to verify their email address by clicking on the confirmation link
    5. Upon successful registration, the user should be redirected to the login page
    """)

# Check if the app should be reset
if st.session_state.get('reset_app', False):
    # Clear the reset flag
    st.session_state.reset_app = False
    # Rerun the app to reset the UI
    st.rerun()

# File upload handling
uploaded_file = st.file_uploader("Upload a .docx, .pdf, or .txt file", type=["docx", "pdf", "txt"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        user_story = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        user_story = extract_text_from_docx(uploaded_file)
    elif file_extension == "txt":
        user_story = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file format")
        user_story = ""

    st.text_area("Enter User Story", value=user_story, height=300)
else:
    user_story = st.text_area(
        "Enter User Story",
        height=300,
        placeholder="Describe your feature or user story in any format. You can include structured information if available, or just describe what the feature should do."
    )

# Set up user interface controls in 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    # Add generation mode as radio buttons
    st.write("**Test Case Generation Mode**")
    generation_mode = st.radio(
        "Mode",
        options=["Comprehensive Coverage", "Specific Number"],
        index=0 if st.session_state.generation_mode == "Comprehensive Coverage" else 1,
        label_visibility="collapsed",
        help="Choose whether to generate comprehensive coverage or a specific number of test cases"
    )

    # Store the generation mode in session state
    st.session_state.generation_mode = generation_mode

    # Show test count input only for "Specific Number" mode
    if generation_mode == "Specific Number":
        test_count = st.number_input(
            "Number of test cases",
            min_value=5,
            max_value=200,
            value=15,
            step=5
        )

        # Add hint for larger test case counts
        if test_count > 50:
            st.info("For large test case counts (>50), generation will be done in batches and may take longer.")
    else:
        # For comprehensive coverage, we don't show the number input
        test_count = 100  # Default high value, used as a parameter but not shown to user

with col2:
    # Add priority dropdown
    st.write("**Test Case Priority**")
    priority_filter = st.selectbox(
        "Priority",
        options=["All", "High", "Medium", "Low"],
        index=0,  # Default to "All"
        label_visibility="collapsed",
        help="Generate test cases of specific priority level only"
    )

with col3:
    # Add spacing to align the button with the other elements
    st.write("&nbsp;")
    # Generate button
    generate_button = st.button("Generate Test Cases", use_container_width=True)

# Generate test cases inside container
with st.container(height=350):
    if generate_button:
        if not user_story or len(user_story) < 20:
            st.error("Please enter a more detailed user story (at least 20 characters)")
        elif not priority_filter:
            st.error("Please select a priority level")
        else:
            st.session_state.has_generated = True
            # Generate based on mode
            if generation_mode == "Specific Number":
                test_cases = generate_and_process_test_cases(user_story, test_count, priority_filter)
            else:
                # For comprehensive mode, test_count is just a placeholder
                test_cases = generate_and_process_test_cases(user_story, priority_filter=priority_filter)

            # Display detailed test cases
            st.header("Detailed Test Cases")
            display_detailed_test_cases(test_cases)

    # Display previously generated results
    elif st.session_state.get('just_stopped', False) or (st.session_state.user_story_test_cases and st.session_state.current_story_id):
        # Reset the flag if it exists
        if 'just_stopped' in st.session_state:
            st.session_state.just_stopped = False

        story_id = st.session_state.current_story_id
        if story_id in st.session_state.user_story_test_cases:
            test_cases = st.session_state.user_story_test_cases[story_id]["test_cases"]

            st.header("Previous Test Cases")
            display_detailed_test_cases(test_cases)

# Show result buttons outside the container
if 'test_cases' in locals() or 'test_cases' in globals():
    show_result_buttons(test_cases)
