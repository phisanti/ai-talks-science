import os
from podcastfy.utils.config import Config
from podcastfy.utils.config_conversation import ConversationConfig
from podcastfy.content_generator import LLMBackend
from langchain.prompts import ChatPromptTemplate
from typing import Optional, Dict, Any, Union

def close_podcast(
    config: Dict[str, Any], 
    instr: str, 
    llm: LLMBackend, 
    paper_info: Optional[Union[str, Dict[str, Any]]] = None
) -> str:
    """
    Generates a closing segment for a podcast based on paper information.
    
    Args:
        config: Configuration dictionary containing podcast parameters
        instr: Instructions for the closing generation
        llm: LLM backend instance for text generation
        paper_info: Paper information to use in the closing. If None, uses config['doctitle']
    
    Returns:
        str: Generated closing text for the podcast
    """
    if paper_info is None:
        paper_info = config.get('doctitle', '...')
    # Create structured prompt
    prompt = ChatPromptTemplate.from_template("""
    PAPER INFORMATION:
    {paper_info}
    
    INSTRUCTIONS:
    {instr}
    """)
    
    # Create modern chain
    chain = prompt | llm.llm
    
    # Run chain
    inputs = {
        "instr": instr,
        "conversation_style": config.get("conversation_style", "casual"),
        "output_language": config.get("output_language", "English"),
        "paper_info": paper_info
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
    introduction = close_podcast(conversation_config.to_dict(), llm)
    print(introduction)
