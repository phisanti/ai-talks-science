from typing import Optional, Dict, Union
import re


def format_podcast_duration(duration_str: str, segment: Optional[str] = None) -> str:
    """
    Format podcast duration guidance with precise word counts and time allocations.
    
    Args:
        duration_str: A string containing the podcast duration (e.g., "5 min", "30 minutes")
        segment: Optional specific segment to get guidance for ("introduction", "qa"/"discussion", "ending"/"conclusion")
    
    Returns:
        Formatted string with podcast duration guidance, word counts, and constraints
        
    Raises:
        ValueError: If the duration format is invalid or duration is non-positive
    """
    # Safety check 1: Validate input format
    if not duration_str or not isinstance(duration_str, str):
        raise ValueError("Duration must be a non-empty string (e.g., '15 min', '30 minutes')")
    
    # Extract the numerical value from the duration string
    duration_match = re.search(r'(\d+)', duration_str)
    if not duration_match:
        raise ValueError("Invalid duration format. Must contain a number (e.g., '15 min')")
    
    duration_value = int(duration_match.group(1))
    
    # Safety check 2: Ensure positive duration
    if duration_value <= 0:
        raise ValueError("Podcast duration must be a positive number of minutes")
    
    # Define duration categories and their corresponding structures
    duration_formats: Dict[int, Dict[str, Union[float, str]]] = {
        5: {
            "total": 700, 
            "intro": 0.3, 
            "discussion": 0.5/3, 
            "conclusion": 0.2,
            "extra": "This is a brief podcast format. Focus on creating a concise overview that captures only the key insights. Synthesize questions and answers to their core essence, avoiding tangents. Prioritize clarity over comprehensiveness and maintain a brisk conversational pace throughout. DO NOT EXCEED THE WORD LIMITS UNDER ANY CIRCUMSTANCES."
        },
        10: {
            "total": 1400, 
            "intro": 0.2, 
            "discussion": 0.6/3, 
            "conclusion": 0.2,
            "extra": "This short-form podcast requires efficient communication. Cover the most important aspects of the paper while maintaining enough detail to be informative. Consolidate similar questions and ensure the discussion remains focused on central themes rather than exploring minor details. THE SPECIFIED WORD COUNTS ARE ABSOLUTE MAXIMUM LIMITS - STAY BELOW THEM."
        },
        15: {
            "total": 2000, 
            "intro": 0.2, 
            "discussion": 0.67/3, 
            "conclusion": 0.13,
            "extra": "This medium-length podcast allows for a balanced approach between brevity and depth. Prioritize the most significant findings and methodologies while providing enough context for listener comprehension. Structure the discussion to build logically with a clear narrative arc. YOU MUST ADHERE STRICTLY TO THE WORD AND TIME CONSTRAINTS PROVIDED."
        },
        20: {
            "total": 3000, 
            "intro": 0.25, 
            "discussion": 0.5/3, 
            "conclusion": 0.25,
            "extra": "This format provides room to explore major concepts with moderate depth. The introduction should properly frame the paper's context and significance, while the conclusion should emphasize practical implications. Maintain engaging pacing throughout the core discussion. THE WORD LIMITS SPECIFIED ARE HARD BOUNDARIES THAT CANNOT BE EXCEEDED."
        },
        30: {
            "total": 4000, 
            "intro": 0.17, 
            "discussion": 0.5/3, 
            "conclusion": 0.33,
            "extra": "This podcast length allows for comprehensive coverage. Use the extended discussion section to explore key methodologies, findings, and implications with appropriate depth. The longer conclusion provides space to consider broader impacts and future directions in the field. STRICTLY CONFORM TO THE WORD COUNTS PROVIDED - THEY ARE NOT SUGGESTIONS."
        },
        45: {
            "total": 6000, 
            "intro": 0.18, 
            "discussion": 0.6/3, 
            "conclusion": 0.22,
            "extra": "This extended format permits in-depth exploration of complex concepts. The substantial discussion section should thoroughly examine methodologies, results, limitations, and connections to existing research. Include thoughtful analogies or examples to help listeners grasp difficult concepts. THE DURATION AND WORD LIMITS ARE INFLEXIBLE CONSTRAINTS THAT MUST BE FOLLOWED EXACTLY."
        },
        60: {
            "total": 8000, 
            "intro": 0.17, 
            "discussion": 0.66/3, 
            "conclusion": 0.17,
            "extra": "This comprehensive podcast format allows for thorough examination of all aspects of the paper. The extensive discussion section should explore nuances, technical details, methodological choices, implications, and connections to the broader field. Incorporate engaging narrative elements while maintaining scientific accuracy and depth. DO NOT EXCEED THE WORD COUNTS UNDER ANY CIRCUMSTANCES - THEY ARE ABSOLUTE LIMITS."
        }
    }
    
    # Hard constraint message to append
    hard_constraint_msg = "THESE LIMITS ARE NON-NEGOTIABLE. You must fit your content within these exact boundaries - never exceed them. Edit ruthlessly to ensure compliance with these strict word counts."
    heading = "DURATIONRULES: This is the "
    # Find the closest duration key
    closest_duration = min(duration_formats.keys(), key=lambda x: abs(x - duration_value))
    format_data = duration_formats[closest_duration]
    
    # Cast to the correct types for type checking
    total_words = int(format_data["total"])  
    intro_ratio = float(format_data["intro"])  
    discussion_ratio = float(format_data["discussion"])  
    conclusion_ratio = float(format_data["conclusion"])  
    extra_info = str(format_data["extra"])  
    
    # If a specific segment is requested, return targeted information
    if segment:
        segment_lower = segment.lower()
        
        if segment_lower == "introduction":
            segment_time = int(closest_duration * intro_ratio)
            segment_words = int(total_words * intro_ratio)
            return f"{heading}Introduction segment: {segment_time} minutes (approximately {segment_words} words). {hard_constraint_msg} {extra_info}"
        
        elif segment_lower in ["qa", "discussion", "body"]:
            segment_time = int(closest_duration * discussion_ratio)
            segment_words = int(total_words * discussion_ratio)
            return f"{heading}Discussion/Q&A segment: {segment_time} minutes (approximately {segment_words} words). {hard_constraint_msg} {extra_info}"
        
        elif segment_lower in ["ending", "conclusion", "close"]:
            segment_time = int(closest_duration * conclusion_ratio)
            segment_words = int(total_words * conclusion_ratio)
            return f"{heading}Conclusion segment: {segment_time} minutes (approximately {segment_words} words). {hard_constraint_msg} {extra_info}"
        
        else:
            # Safety check: Handle invalid segment
            return f"Unknown segment '{segment}'. Valid options are 'introduction', 'discussion'/'qa', or 'conclusion'/'ending'."
    
    # Return full format if no segment specified
    intro_time = int(closest_duration * intro_ratio)
    discussion_time = int(closest_duration * discussion_ratio)
    conclusion_time = int(closest_duration * conclusion_ratio)
    intro_words = int(total_words * intro_ratio)
    discussion_words = int(total_words * discussion_ratio)
    conclusion_words = int(total_words * conclusion_ratio)
    
    return (
        f"{heading}{duration_value} minutes podcast (approximately {total_words} words total):\n"
        f"- Introduction: {intro_time} min (~{intro_words} words)\n"
        f"- Discussion: {discussion_time} min (~{discussion_words} words)\n"
        f"- Conclusion: {conclusion_time} min (~{conclusion_words} words)\n\n"
        f"{hard_constraint_msg} {extra_info}"
    )

