import re
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Callable, Union

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_test_case_output(text: str, format_type: str = "tabular") -> List[Dict[str, Any]]:
    """
    Parse LLM output into structured test cases
    
    Args:
        text: Raw text output from LLM
        format_type: "tabular" for tabular format with | separators, or "freeform" for section-based parsing
    
    Returns:
        List of test case dictionaries
    """
    if format_type == "tabular":
        return parse_tabular_output(text)
    else:
        return parse_freeform_output(text)

def parse_tabular_output(text: str) -> List[Dict[str, Any]]:
    """
    Parse tabular output into test cases with improved column handling
    
    Args:
        text: Raw text output in tabular format with | separators
        
    Returns:
        List of test case dictionaries
    """
    test_cases = []
    
    # Split by lines to process rows
    lines = text.strip().split('\n')
    header_found = False
    column_indices = {}
    
    # Pre-process to clean up any markdown table formatting issues
    cleaned_lines = []
    for line in lines:
        # Skip separator lines (---|---|---)
        if re.match(r'^[\s\-\|]+$', line):
            continue
        # Keep content lines
        if "|" in line:
            cleaned_lines.append(line)
    
    # Process each line
    for line in cleaned_lines:
        if not line.strip() or "|" not in line:
            continue
            
        # Split the line by pipe symbol
        cells = [cell.strip() for cell in line.split('|')]
        
        # Remove empty cells at beginning/end
        if not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
            
        # If too few columns, skip
        if len(cells) < 3:
            continue
            
        # Check if this is a header row
        if not header_found and any(x in line.lower() for x in ["test id", "testid", "tc_", "test case"]) and any(x in line.lower() for x in ["priority", "feature", "scenario"]):
            header_found = True
            
            # Map column names to indices
            for idx, header in enumerate(cells):
                header_lower = header.lower().strip()
                if "test id" in header_lower or "testid" in header_lower or header_lower.startswith("tc_"):
                    column_indices["test_id"] = idx
                elif "feature" in header_lower:
                    column_indices["feature"] = idx
                elif "scenario" in header_lower:
                    column_indices["scenario"] = idx
                elif "description" in header_lower:
                    column_indices["description"] = idx
                elif "precondition" in header_lower:
                    column_indices["precondition"] = idx
                elif "test step" in header_lower or "steps" in header_lower:
                    column_indices["steps"] = idx
                elif "expected" in header_lower or "results" in header_lower:
                    column_indices["expected"] = idx
                elif "priority" in header_lower:
                    column_indices["priority"] = idx
            continue
            
        # Process data rows - more flexible matching for test IDs
        if ("TC_" in line or "tc_" in line.lower() or re.search(r'\bTC\d+\b', line)) and "|" in line:
            # If we found a header, use the column indices
            if header_found and column_indices and len(column_indices) >= 5:  # Need at least 5 columns to be valid
                tc = {}
                for field, idx in column_indices.items():
                    if idx < len(cells):
                        tc[field] = cells[idx]
                    else:
                        tc[field] = ""
                
                test_case = {
                    "Test ID": tc.get("test_id", ""),
                    "Feature": tc.get("feature", ""),
                    "Scenario": tc.get("scenario", ""),
                    "Description": tc.get("description", ""),
                    "Precondition": tc.get("precondition", ""),
                    "Test Steps": tc.get("steps", ""),
                    "Expected Results": tc.get("expected", ""),
                    "Priority": tc.get("priority", "Medium")
                }
            else:
                # If no header found or header is incomplete, assume standard order
                # This is a fallback for when header mapping fails
                test_case = {
                    "Test ID": cells[0] if len(cells) > 0 else "",
                    "Feature": cells[1] if len(cells) > 1 else "",
                    "Scenario": cells[2] if len(cells) > 2 else "",
                    "Description": cells[3] if len(cells) > 3 else "",
                    "Precondition": cells[4] if len(cells) > 4 else "",
                    "Test Steps": cells[5] if len(cells) > 5 else "",
                    "Expected Results": cells[6] if len(cells) > 6 else "",
                    "Priority": cells[8] if len(cells) > 8 else "Medium"
                }
            
            # Validate and fix priority field - ensure it's "High", "Medium", or "Low"
            priority = test_case.get("Priority", "").strip().lower()
            if priority not in ["high", "medium", "low"]:
                # Attempt to extract priority from the value
                if "high" in priority:
                    test_case["Priority"] = "High"
                elif "low" in priority:
                    test_case["Priority"] = "Low"
                elif "medium" in priority or "med" in priority:
                    test_case["Priority"] = "Medium"
                else:
                    # Default to Medium
                    test_case["Priority"] = "Medium"
            else:
                # Normalize capitalization
                test_case["Priority"] = priority.capitalize()
            
            # Only add if we have a valid test ID and haven't seen it before
            if test_case["Test ID"]:
                # Check if test case with this ID already exists
                existing_ids = [tc["Test ID"] for tc in test_cases]
                if test_case["Test ID"] not in existing_ids:
                    test_cases.append(test_case)
    
    return test_cases

def parse_freeform_output(raw_output: str) -> List[Dict[str, Any]]:
    """
    Parse the raw LLM output into structured test cases with improved format detection.
    This is optimized for freeform/section-based test case descriptions.
    
    Args:
        raw_output: The raw text output from the LLM
        
    Returns:
        List of dictionaries containing structured test case data
    """
    structured_test_cases = []
    
    try:
        # Split output by test cases using a more robust pattern
        # Look for TC_XXX patterns with test objectives
        test_case_pattern = r'(?:^|\n)(?:\*\*)?(?:TC_\d+)(?:\s+|\s*:\s*)(.*?)(?=(?:\n(?:\*\*)?TC_\d+)|$)'
        matches = re.findall(test_case_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        
        if matches:
            test_case_blocks = re.split(r'(?:^|\n)(?:\*\*)?TC_\d+(?:\s+|\s*:\s*)', raw_output)
            # First element is empty or intro text, remove it
            if test_case_blocks and not re.search(r'TC_\d+', test_case_blocks[0]):
                test_case_blocks = test_case_blocks[1:]
        else:
            # Fallback to --- separator if TC pattern doesn't work
            if "---" in raw_output:
                test_case_blocks = re.split(r'\n\s*-{3,}\s*\n', raw_output)
            else:
                # Last resort - try section-based split
                test_case_blocks = [raw_output]  # Just use the whole text as one block
        
        # Extract TC_IDs separately
        tc_ids = re.findall(r'(?:^|\n)(?:\*\*)?(TC_\d+)', raw_output, re.MULTILINE)
        
        # Process each test case block with its corresponding ID
        for i, block in enumerate(test_case_blocks):
            if not block.strip():
                continue
                
            test_case = {}
            
            # Assign the correct TC_ID if available
            if i < len(tc_ids):
                test_case['Test ID'] = tc_ids[i]
            else:
                test_case['Test ID'] = f"TC_{i+1:03d}"  # Fallback ID
            
            # Extract Objective/Description
            obj_match = re.search(r'(?:Test Objective|Description|Verify):\s*(.*?)(?:\n\s*\*|\n\s*Preconditions:|\n\s*Test Steps:)', block, re.DOTALL | re.IGNORECASE)
            if not obj_match:
                # Try another pattern
                obj_match = re.search(r'^(.*?)(?:\n\s*\*|\n\s*Preconditions:|\n\s*Test Steps:)', block, re.DOTALL | re.IGNORECASE)
            
            if obj_match:
                test_case['Description'] = obj_match.group(1).strip()
            else:
                # If still not found, just take the first non-empty line
                lines = [line.strip() for line in block.split('\n') if line.strip()]
                test_case['Description'] = lines[0] if lines else "Unknown objective"
            
            # Set Feature and Scenario based on Description
            # This is a simplification since freeform text might not have these fields
            test_case['Feature'] = extract_feature_from_description(test_case['Description'])
            test_case['Scenario'] = extract_scenario_from_description(test_case['Description'])
            
            # Extract Preconditions
            pre_match = re.search(r'Preconditions:?(.*?)(?:Test Steps:|Steps:|$)', block, re.DOTALL | re.IGNORECASE)
            if pre_match:
                preconditions_text = pre_match.group(1).strip()
                # Extract bullet points
                bullet_points = re.findall(r'(?:^|\n)\s*[•\-*]\s*(.*?)(?:\n|$)', preconditions_text)
                if bullet_points:
                    test_case['Precondition'] = '; '.join(bullet_points)
                else:
                    # If no bullet points, try to extract lines
                    lines = [line.strip() for line in preconditions_text.split('\n') if line.strip()]
                    if lines:
                        test_case['Precondition'] = '; '.join(lines)
                    else:
                        test_case['Precondition'] = preconditions_text
            else:
                test_case['Precondition'] = "System is available"
            
            # Extract Test Steps
            steps_match = re.search(r'Test[_ ]?Steps:?(.*?)(?:Expected[_ ]?Results:|$)', block, re.DOTALL | re.IGNORECASE)
            if steps_match:
                steps_text = steps_match.group(1).strip()
                # Parse numbered steps
                steps = re.findall(r'(?:^|\n)\s*\d+\.?\s*(.*?)(?:\n\s*\d+\.|\n\s*Expected|$)', steps_text, re.DOTALL)
                if steps:
                    test_case['Test Steps'] = '; '.join([s.strip() for s in steps])
                else:
                    # Try another approach to extract steps
                    steps = [line.strip() for line in steps_text.split('\n') if line.strip()]
                    if steps:
                        test_case['Test Steps'] = '; '.join(steps)
                    else:
                        test_case['Test Steps'] = steps_text
            else:
                test_case['Test Steps'] = "Follow the test procedure"
            
            # Extract Expected Results
            results_match = re.search(r'Expected[_ ]?Results:?(.*?)(?:Test[_ ]?Data:|Priority:|$)', block, re.DOTALL | re.IGNORECASE)
            if results_match:
                results_text = results_match.group(1).strip()
                # Try to extract bullet points
                bullet_results = re.findall(r'(?:^|\n)\s*[•\-*]\s*(.*?)(?:\n|$)', results_text)
                if bullet_results:
                    test_case['Expected Results'] = '; '.join([r.strip() for r in bullet_results])
                else:
                    # If no bullet points, just use the text as is
                    test_case['Expected Results'] = results_text
            else:
                # Try to find expected results in another way
                results_block = re.search(r'(?:^|\n)Expected Results:?\s*(.*?)(?:$)', block, re.DOTALL | re.IGNORECASE)
                if results_block:
                    results_text = results_block.group(1).strip()
                    test_case['Expected Results'] = results_text
                else:
                    test_case['Expected Results'] = "Test passes if all expected conditions are met"
            
            # Extract Priority
            priority_match = re.search(r'Priority:?\s*(.*?)(?:\n|$)', block, re.IGNORECASE)
            if priority_match:
                priority_text = priority_match.group(1).strip().lower()
                if "high" in priority_text:
                    test_case['Priority'] = "High"
                elif "low" in priority_text:
                    test_case['Priority'] = "Low"
                elif "medium" in priority_text or "med" in priority_text:
                    test_case['Priority'] = "Medium"
                else:
                    test_case['Priority'] = "Medium"  # Default
            else:
                test_case['Priority'] = "Medium"  # Default priority
            
            structured_test_cases.append(test_case)
        
        return structured_test_cases
        
    except Exception as e:
        logger.error(f"Error parsing test cases: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def extract_feature_from_description(description: str) -> str:
    """
    Extract a feature name from the test case description
    
    Args:
        description: The test case description
        
    Returns:
        A short feature name
    """
    # Try to identify a feature by looking for common phrases
    feature_patterns = [
        r'(?:verify|test|validate)\s+(?:the\s+)?([a-z0-9\s]+?)\s+(?:function|feature|capability)',
        r'(?:^|\s)([a-z0-9\s]+?)\s+functionality'
    ]
    
    for pattern in feature_patterns:
        match = re.search(pattern, description.lower())
        if match:
            return match.group(1).strip().title()
    
    # If no match, extract first few words (up to 5) as the feature
    words = description.split()
    return ' '.join(words[:min(5, len(words))]).title()

def extract_scenario_from_description(description: str) -> str:
    """
    Extract a scenario name from the test case description
    
    Args:
        description: The test case description
        
    Returns:
        A short scenario name
    """
    # Try to identify specific test scenarios
    scenario_patterns = [
        r'with\s+([a-z0-9\s]+)',
        r'when\s+([a-z0-9\s]+)',
        r'for\s+([a-z0-9\s]+)'
    ]
    
    for pattern in scenario_patterns:
        match = re.search(pattern, description.lower())
        if match:
            return match.group(1).strip().title()
    
    # If no match, use the description as is, but limit length
    if len(description) > 50:
        return description[:50].strip() + "..."
    return description

def format_test_case_fields(tc: Dict[str, Any]) -> None:
    """
    Format test case fields for better readability.
    This function operates on the dictionary in-place.
    
    Args:
        tc: Test case dictionary to format
    """
    # Format fields with semicolons as bullet points
    for field in ["Precondition", "Expected Results"]:
        if isinstance(tc.get(field), str) and ";" in tc[field]:
            items = [item.strip() for item in tc[field].split(";") if item.strip()]
            tc[field] = "\n".join([f"• {item}" for item in items])
        
    # Format test steps
    if isinstance(tc.get("Test Steps"), str):
        if ";" in tc["Test Steps"]:
            # Split by semicolons and format as numbered list
            steps = [step.strip() for step in tc["Test Steps"].split(";") if step.strip()]
            
            # Check if already numbered
            if all(re.match(r'^\d+\.?\s', step) for step in steps):
                tc["Test Steps"] = "\n".join(steps)
            else:
                tc["Test Steps"] = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])

def filter_test_cases_by_priority(test_cases: List[Dict[str, Any]], priority: str, adaptive: bool = True) -> List[Dict[str, Any]]:
    """
    Filter test cases to include only those matching the requested priority.
    With adaptive mode, will convert priorities instead of filtering if no matches found.
    
    Args:
        test_cases: List of test case dictionaries
        priority: Priority to filter by ("High", "Medium", "Low", or "All")
        adaptive: If True, converts priorities when no matches found
        
    Returns:
        List of filtered test cases
    """
    if priority == "All":
        return test_cases
    
    # First try with strict filtering
    filtered_cases = []
    original_count = len(test_cases)
    
    for tc in test_cases:
        # Normalize the priority value
        tc_priority = str(tc.get("Priority", "")).strip().lower()
        
        # Check if it matches the requested priority
        if (priority.lower() in tc_priority) or (
            # Handle variations
            (priority.lower() == "high" and "high" in tc_priority) or
            (priority.lower() == "medium" and ("medium" in tc_priority or "med" in tc_priority)) or
            (priority.lower() == "low" and "low" in tc_priority)
        ):
            # Standardize the priority to ensure consistency
            tc["Priority"] = priority  # Use the exact priority string passed in
            filtered_cases.append(tc)
    
    # If we found some matching cases, return them
    if filtered_cases:
        return filtered_cases
        
    # If adaptive mode is enabled and we found no matches, convert priorities instead
    if adaptive and not filtered_cases and test_cases:
        # Log this conversion
        logger.info(f"No test cases with {priority} priority found. Converting existing cases.")
        
        # Just use all cases but convert their priorities
        for tc in test_cases:
            # IMPORTANT: Use the exact priority string, not a normalized version
            tc["Priority"] = priority
        
        return test_cases
    
    # Return empty list if no matches and not adaptive
    return filtered_cases

def create_test_case_dataframe(test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert test cases to a pandas DataFrame for display or export
    
    Args:
        test_cases: List of test case dictionaries
        
    Returns:
        DataFrame containing the test cases
    """
    if not test_cases:
        # Create empty DataFrame with correct columns
        return pd.DataFrame(columns=[
            "Test ID", "Feature", "Scenario", "Description", 
            "Precondition", "Test Steps", "Expected Results", "Priority"
        ])
    
    df = pd.DataFrame(test_cases)
    
    # Ensure all columns exist
    for col in ["Test ID", "Feature", "Scenario", "Description", 
                "Precondition", "Test Steps", "Expected Results", "Priority"]:
        if col not in df.columns:
            df[col] = ""
    
    # Format text fields
    for col in df.columns:
        df[col] = df[col].apply(lambda x: 
            str(x)[:500] + "..." if isinstance(x, str) and len(str(x)) > 500 else str(x))
    
    return df

# Import traceback module which is needed for error handling in parse_freeform_output
import traceback