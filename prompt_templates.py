import re
from typing import List, Dict, Any, Optional, Tuple

def create_test_case_prompt(
    user_story: str, 
    test_count: int, 
    actor: str, 
    action: str, 
    purpose: str, 
    req_text: str, 
    crit_text: str, 
    start_id: int = 1, 
    focus_area: Optional[str] = None, 
    priority_filter: str = "All"
) -> str:
    """
    Create a prompt for test case generation that emphasizes the exact count requirement
    and filters by priority if specified.
    
    Args:
        user_story: The user story text
        test_count: Number of test cases to generate
        actor: Primary actor in the user story
        action: Main action being performed
        purpose: Purpose of the user story
        req_text: Requirements text
        crit_text: Acceptance criteria text
        start_id: Starting ID number for test cases
        focus_area: Optional text to define test focus areas
        priority_filter: Priority filter ("All", "High", "Medium", "Low")
        
    Returns:
        Formatted prompt string for the LLM
    """
    
    # Add priority filter instruction
    priority_instruction = ""
    if priority_filter != "All":
        priority_instruction = f"""
        **EXTREMELY IMPORTANT PRIORITY FILTER INSTRUCTION:**
        - CRUCIAL REQUIREMENT: Generate ONLY test cases with {priority_filter} priority
        - The Priority column for EVERY test case MUST EXACTLY contain: "{priority_filter}"
        - DO NOT include ANY test cases with priorities other than "{priority_filter}"
        - If you start to generate a test case and realize it doesn't warrant a {priority_filter} priority, 
        discard it and generate a different one that does qualify as {priority_filter} priority
        - For clarity: this means NO Low or Medium priority test cases should appear if High is requested,
        NO High or Low priority test cases should appear if Medium is requested, etc.
        """
    else:
        priority_instruction = """
        **PRIORITY INSTRUCTION:**
        - Each test case must have a priority of either "High", "Medium", or "Low"
        - Assign priorities based on the criticality of the test case:
        * High: Core functionality, security, data integrity
        * Medium: Important features, edge cases, performance
        * Low: UI refinements, optional features, rare scenarios
        """
    
    # Additional priority formatting instruction to ensure clarity
    priority_format = """
    **CRITICAL PRIORITY FORMAT INSTRUCTION:**
    - The Priority column MUST ONLY contain one of these three exact values: "High", "Medium", or "Low"
    - No other words or variations are allowed in the Priority column
    - Do not mix the Priority column with other information
    """
    
    # Focus area text based on where we are in the generation process
    if not focus_area:
        focus_area = """
        1. Positive scenarios (normal flows)
        2. Negative scenarios (error handling)
        3. Boundary conditions and edge cases
        4. Performance considerations 
        5. Security testing scenarios
        6. Accessibility testing
        7. Cross-browser/device compatibility
        8. Different user roles and permissions
        9. Data validation for each input field
        10. System integrations
        11. Workflow variations
        12. Error message validation
        13. State transitions
        14. Concurrency and multi-user scenarios
        15. Localization/internationalization aspects
        """
    
    # Strengthen the emphasis on exact test count
    count_emphasizer = f"""
    **EXTREMELY IMPORTANT: YOU MUST GENERATE EXACTLY {test_count} TEST CASES**
    - The most critical requirement is to generate EXACTLY {test_count} test cases, no more and no less
    - Each test must have a unique ID from TC_{start_id:03d} to TC_{start_id+test_count-1:03d}
    - Do not stop generating until you have produced all {test_count} test cases
    - Do not include any explanatory text before or after the test cases
    - Start immediately with the table header and then the test cases
    """
    
    prompt = f"""
You are a highly advanced test case generator model specialized in quality assurance. 

{count_emphasizer}

**User Story:**
{user_story}

**Additional Context:**
- Primary Actor: {actor}
- Main Action: {action}
- Purpose: {purpose}
- Requirements: {req_text if req_text else "Complete the functionality described in the user story"}
- Acceptance Criteria: {crit_text if crit_text else "Ensure the system works as described in the user story"}
{priority_instruction}
{priority_format}

**Required Output Format (Tabular Structure with | separators):**

Test ID | Feature | Scenario | Description | Precondition | Test Steps | Expected Results | Priority
--------|---------|----------|-------------|--------------|------------|------------------|----------

**Format Instructions:**
- Test ID: Use TC_XXX format (TC_{start_id:03d} through TC_{start_id+test_count-1:03d})
- Feature: Brief name of feature being tested (extract from user story)
- Scenario: Specific test scenario name
- Description: Clear explanation of what's being tested
- Precondition: List setup requirements (separate multiple items with semicolons)
- Test Steps: Numbered steps (e.g., "1. Login; 2. Navigate to dashboard; 3. Click create button")
- Expected Results: Expected outcome for each step (separate with semicolons)
- Priority: ONLY use "High", "Medium", or "Low" - nothing else in this column

**Test Coverage Guidance to Help You Generate {test_count} Test Cases:**
{focus_area}

**REMINDER: YOUR RESPONSE MUST CONTAIN EXACTLY {test_count} TEST CASES IN THE TABULAR FORMAT.**

**Example Row:**
TC_{start_id:03d} | User Login | Valid Credentials | Verify user can login with valid username and password | Application is accessible; User has valid account | 1. Navigate to login page; 2. Enter valid username; 3. Enter valid password; 4. Click Login button | Login successful; User redirected to dashboard; Welcome message displays user name | {priority_filter if priority_filter != "All" else "High"}

Remember: If you find yourself struggling to create unique test cases, consider more granular variations of scenarios, different data combinations, and all possible user interactions mentioned or implied in the story.

Your response should begin immediately with the table header followed by the {test_count} test cases.
"""
    return prompt

def get_focus_areas_for_chunk(start_index: int, total_count: int) -> str:
    """
    Return specific focus areas for a chunk to avoid repetitive test cases
    
    Args:
        start_index: Starting index of the current chunk
        total_count: Total number of test cases to generate
        
    Returns:
        Focus area prompt string
    """
    # Calculate what percentage through the total we are
    progress = start_index / total_count if total_count > 0 else 0
    
    if progress < 0.2:
        return """
        For this batch of test cases, focus on:
        - Core functionality (happy path scenarios)
        - Basic input validation
        - Main user workflows
        - Primary use cases mentioned in the user story
        """
    elif progress < 0.4:
        return """
        For this batch of test cases, focus on:
        - Edge cases and boundary values
        - Error handling scenarios
        - Invalid inputs and data validation
        - Different user roles and permissions
        """
    elif progress < 0.6:
        return """
        For this batch of test cases, focus on:
        - Performance considerations
        - Security testing scenarios
        - Integration points with other systems
        - Non-functional requirements
        """
    elif progress < 0.8:
        return """
        For this batch of test cases, focus on:
        - Accessibility testing
        - Different browsers and devices
        - Localization/internationalization
        - Database interactions and data persistence
        """
    else:
        return """
        For this batch of test cases, focus on:
        - Unusual user behaviors
        - Recovery from errors or interruptions
        - Extreme edge cases
        - System-level interactions
        - Any remaining scenarios not covered in previous batches
        """

def extract_user_story_components(user_story: str) -> Tuple[str, str, str, str, str]:
    """
    Extract components from a user story text: actor, action, purpose, requirements, criteria
    
    Args:
        user_story: The user story text
        
    Returns:
        Tuple containing (actor, action, purpose, req_text, crit_text)
    """
    # Default values
    actor = "User"
    action = "perform the described functionality"
    purpose = "accomplish their goals"
    
    # Try to extract from standard format
    actor_match = re.search(r'[Aa]s (?:an?|the) ([^,]*)', user_story)
    if actor_match:
        actor = actor_match.group(1).strip()
    
    action_match = re.search(r'I want to ([^,\.]*)', user_story)
    if action_match:
        action = action_match.group(1).strip()
    
    purpose_match = re.search(r'so that ([^,\.]*)', user_story)
    if purpose_match:
        purpose = purpose_match.group(1).strip()
    
    # Extract requirements and criteria
    requirements = []
    criteria = []
    
    # Look for numbered lists
    items = re.findall(r'\d+\.\s*(.*?)(?=\d+\.|$)', user_story, re.DOTALL)
    if items:
        # Split items between requirements and criteria
        middle = len(items) // 2
        requirements = items[:middle]
        criteria = items[middle:]
    
    # Look for sections labeled "Requirements" or "Acceptance Criteria"
    req_section = re.search(r'(?:Requirements|Functional Perspective):(.*?)(?:Acceptance Criteria:|$)', user_story, re.DOTALL | re.IGNORECASE)
    if req_section:
        req_items = re.findall(r'(?:^|\n)\s*[•\-*]?\s*(\d+\.?|\w+\.|\*)\s*(.*?)(?:\n|$)', req_section.group(1))
        if req_items:
            requirements.extend([item[1].strip() for item in req_items])
    
    crit_section = re.search(r'Acceptance Criteria:(.*?)(?:$)', user_story, re.DOTALL | re.IGNORECASE)
    if crit_section:
        crit_items = re.findall(r'(?:^|\n)\s*[•\-*]?\s*(\d+\.?|\w+\.|\*)\s*(.*?)(?:\n|$)', crit_section.group(1))
        if crit_items:
            criteria.extend([item[1].strip() for item in crit_items])
    
    # Format the output
    req_text = "\n".join([f"- {req.strip()}" for req in requirements])
    crit_text = "\n".join([f"- {crit.strip()}" for crit in criteria])
    
    return actor, action, purpose, req_text, crit_text

def get_optimal_temperature(test_count: int, chunk_number: int = 1, attempt: int = 0) -> float:
    """
    Determine optimal temperature setting based on test count and generation attempt
    
    Args:
        test_count: Number of test cases to generate
        chunk_number: Current chunk number for multi-chunk generations 
        attempt: Current attempt number (for retry logic)
        
    Returns:
        Optimal temperature value
    """
    # Base temperature depends on test count
    if test_count <= 20:
        base_temp = 0.2  # Low temperature for more predictable results
    elif test_count <= 50:
        base_temp = 0.4  # Medium temperature for more variation
    else:
        base_temp = 0.7  # Higher temperature for maximum creativity
    
    chunk_adjustment = min(0.1 * (chunk_number - 1), 0.2)
    attempt_adjustment = min(0.2 * attempt, 0.5)
    
    # Calculate final temperature, capped at 0.95
    final_temp = min(0.95, base_temp + chunk_adjustment + attempt_adjustment)
    
    return round(final_temp, 2)