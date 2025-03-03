"""
Podcast Introduction Generator Module.

This module provides functionality to extract metadata from academic papers
and generate engaging podcast introductions based on that information.
It uses LLM-based content generation to create natural-sounding podcast
openings that accurately represent the paper's content.

Security note: This module interfaces with external LLM APIs and requires
proper API key management through environment variables.
"""
import os
from typing import Dict, Any, Union
from langchain.prompts import ChatPromptTemplate

from podcastfy.content_parser.content_extractor import PDFExtractor
from podcastfy.utils.config import Config
from podcastfy.utils.config_conversation import ConversationConfig
from podcastfy.content_generator import LLMBackend
from podcastfy.template_reader import TemplateLoader


def extract_info(file_path: str, llm: LLMBackend) -> str:
    """
    Extract metadata from the first page of a PDF paper.
    
    This function processes the first page of an academic paper to extract
    key metadata including authors, title, abstract, and affiliations.
    
    Args:
        file_path: Path to the PDF file
        llm: Language model backend for extraction
        
    Returns:
        Structured paper information as text
        
    Raises:
        ValueError: If PDF extraction fails or content cannot be processed
    """
    try:    
        # Get PDF content
        pdf_extractor = PDFExtractor()
        content = pdf_extractor.extract_page1(file_path)
        
        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_template("""
        Extract the following information from this academic paper content.
        
        CONTENT:
        {content}
        
        Please extract and return only the following information about the paper:
        - paper_authors: A list of all authors with their institution references
        - paper_title: The complete title of the paper
        - paper_abstract: The complete abstract of the paper
        - affiliations: The full affiliations of each author with institution
        
        Format your response as structured data with these exact field names.
        If information is not available, leave the field empty. Do not invent information.
        """)
        
        # Create modern chain using |
        chain = prompt | llm.llm
        
        # Run chain with proper input format
        result = chain.invoke({"content": content})
        
        return result.content
    
    except Exception as e:
        raise ValueError(f"Error extracting PDF content: {str(e)}")

def generate_introduction(
    paper_info: Union[str, Dict[str, Any]], 
    instr: str, 
    config: Dict[str, Any], 
    llm: LLMBackend
) -> str:
    """
    Generate an engaging podcast introduction based on paper information.
    
    This function creates a natural-sounding podcast introduction that
    contextualizes the paper and sets up the conversation that follows.
    
    Args:
        paper_info: Extracted paper information (structured text or dict)
        instr: Instructions for the introduction generation
        config: Configuration dictionary with style parameters
        llm: Language model backend for generation
        
    Returns:
        Formatted introduction text for the podcast
        
    Raises:
        ValueError: If the generation fails or produces invalid content
    """
    
    # Create structured prompt
    prompt = ChatPromptTemplate.from_template("""
    PAPER INFORMATION:
    {paper_info}
    INSTRUCTIONS:
    {instr}
    """)
    
    # Create chain
    chain = prompt | llm.llm
    
    # Run chain
    inputs = {
        "paper_info": paper_info,
        "instr": instr,
        "conversation_style": config.get("conversation_style", "casual"),
        "output_language": config.get("output_language", "English")
    }
    
    result = chain.invoke(inputs)
    return result.content

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
    model_name = "gemini-1.5-flash-latest"
    api_key_label = "GEMINI_API_KEY"
    
    # Initialize LLM
    llm = LLMBackend(
        is_local=False, 
        temperature=0.7, 
        max_output_tokens=2048, 
        model_name=model_name, 
        api_key_label=api_key_label
    )
    
    # Create processing pipeline
    paper_info = extract_info(file_path, llm)
    reader = TemplateLoader()
    instructions = reader.get_template("./podcastfy/configs/instructions_introduction.md")
    instructions = reader.fill_template(instructions, config)
    introduction = generate_introduction(paper_info, instructions, conversation_config.to_dict(), llm)
    print(introduction)
