import os
import time
import random
from podcastfy.content_parser.content_extractor import PDFExtractor, ContentExtractor
from podcastfy.utils.config import Config
from podcastfy.utils.config_conversation import ConversationConfig
from podcastfy.content_generator import LLMBackend
from podcastfy.template_reader import TemplateLoader
import google.generativeai as genai
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any

def generate_section_info(section_name: str, context: str, template_reader: TemplateLoader, llm: LLMBackend) -> Dict[str, Any]:
    """Generate information for a specific section of the paper."""
    section_instr = template_reader.get_sections(
        "generate_qa", 
        [section_name, "TASK:", "REQUIREMENTS:"], 
        remove_hashtags=True
    )
    
    # Create structured prompt
    prompt = ChatPromptTemplate.from_template("""
    PAPER INFORMATION:
    {context}
    
    CONVERSATION TEMPLATE:
    {section_instr}
    """)
    
    # Create chain
    chain = prompt | llm.llm
    
    # Run with retries
    max_attempts = 3
    attempts = 0
    
    while attempts < max_attempts:
        try:
            # Invoke chain
            result = chain.invoke({
                "context": context,
                "section_instr": section_instr
            })
            return result.content
            
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts} failed for {section_name}: {str(e)}")
            if attempts >= max_attempts:
                raise ValueError(f"Failed to generate valid QA information for {section_name} after {max_attempts} attempts: {e}")
    
    raise ValueError(f"Failed to generate valid QA information for {section_name}")


def generate_qainformation(context: str, 
                           template_reader, 
                           #conversation_config: Dict[str, Any], 
                           llm: LLMBackend) -> Dict[str, Any]:
    """Extracts content from paper and generates structured QA information."""
    
    # Define sections to process
    sections = [
        "1. Background",
        "2. MainContributions", 
        "3. Limitations"
    ]
    
    # Process each section and combine results
    combined_data = {}
    for section in sections:
        try:
            section_key = section.split()[1].lower()  # Extract "background", "maincontribution", "limitations"
            section_data = generate_section_info(section, context, template_reader, llm)
            combined_data[section_key] = section_data
        except Exception as e:
            print(f"Error processing section {section}: {str(e)}")
    
    # Ensure we have at least some data
    if not combined_data:
        raise ValueError("Failed to generate any valid QA information")
        
    return combined_data



def generate_qasession(qa_result: Dict[str, Any], qa_template, conversation_config: Dict[str, Any], llm: LLMBackend) -> Dict[str, Any]:
    
    # Extract conversation style and language from config

    conversation_style = conversation_config.get("style", "casual and informative")
    output_language = conversation_config.get("language", "English")
    person1_name = conversation_config.get("person1_name", "Interviewer")
    person2_name = conversation_config.get("person2_name", "Guest")
 
    print(qa_template)
    # Create a prompt template for generating conversation segments
    conversation_prompt = ChatPromptTemplate.from_template("""
        QA INFORMATION:
        {section_content}

        SECTION TYPE: {section_name}

        QATEMPLATE:
        {qa_template}
        
        INTERVIEWERNAME: {person1_name}
        GUESTNAME: {person2_name}
        """)
    
    # Create the conversation generation chain
    conversation_chain = (
        RunnablePassthrough() 
        | conversation_prompt 
        | llm.llm 
        | StrOutputParser()
    )
    # Section order for a logical conversation flow
    section_order = ["background", "maincontributions", "limitations"]

    # Generate conversation segments for each section
    conversation_segments = []
    for section_name in section_order:
        if section_name in qa_result:
            # Retry logic for rate limiting
            max_retries = 3
            retry_count = 0
            base_wait_time = 10  # seconds
            
            while retry_count < max_retries:
                try:
                    segment = conversation_chain.invoke({
                        "section_content": qa_result[section_name],
                        "section_name": section_name.capitalize(),
                        "qa_template": qa_template,
                        "conversation_style": conversation_style,
                        "output_language": output_language,
                        "person1_name": person1_name,
                        "person2_name": person2_name
                    })
                    conversation_segments.append(segment)
                    
                    # Add a small delay between sections to avoid rate limiting
                    if section_name != section_order[-1]:
                        time.sleep(2 + random.random())
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e).lower()
                    
                    # Check if it's likely a rate limit error
                    is_rate_limit = any(term in error_msg for term in 
                                       ["rate limit", "ratelimit", "too many requests", 
                                        "429", "quota exceeded", "throttle"])
                    
                    if is_rate_limit or retry_count < max_retries:
                        # Exponential backoff with jitter
                        wait_time = base_wait_time * (2 ** (retry_count - 1)) + random.uniform(1, 5)
                        print(f"Rate limit reached or error occurred. Waiting {wait_time:.2f} seconds before retry {retry_count}/{max_retries}...")
                        time.sleep(wait_time)
                    else:
                        print(f"Error generating conversation for section {section_name}: {str(e)}")
                        raise
    
    # Combine all segments into a coherent conversation
    #if not conversation_segments:
    #    raise ValueError("Failed to generate any conversation segments")
    
    # Join segments, ensuring smooth transitions
    full_conversation = "\n\n".join(conversation_segments)
    
    return full_conversation

if __name__ == "__main__":
    PROJECT_PATH = "./projects/project_1"
    INPUTS = {
        "pdf_path": os.path.join(PROJECT_PATH, "2412.18925v1.pdf"),
        "transcript_path": os.path.join(PROJECT_PATH, "doc_transcript.txt"),
        "base_config": "./podcastfy/configs/config.yaml",
        "conversation_config": os.path.join(PROJECT_PATH, "conversation_config.yaml")
    }

    # Load configs
    config = Config(INPUTS["base_config"])
    conversation_config = ConversationConfig(path=INPUTS["conversation_config"])

    file_path = INPUTS["pdf_path"]
    model_name = "gemini-2.0-flash"
    api_key_label = "GEMINI_API_KEY"
    import json
    import ast
    def load_paper_info(file_path):
        """Load paper info from a text file, handling different formats"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            # Try parsing as JSON first
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                # Try parsing as Python literal
                return ast.literal_eval(content)
            except (SyntaxError, ValueError):
                # If neither works, return as plain text
                return {"raw_content": content}

    # Initialize LLM
    llm = LLMBackend(
        is_local=False, 
        temperature=0.7, 
        max_output_tokens=2**15, 
        model_name=model_name, 
        api_key_label=api_key_label,
        structured_response=False
    )
    #llm =None
    # Create processing pipeline
    #paper_info = generate_qainformation(file_path, conversation_config.to_dict(), llm)
    paper_info = load_paper_info("./projects/project_1/testing.txt")
    introduction = generate_qasession(paper_info, conversation_config.to_dict(), llm)
    print(introduction)
