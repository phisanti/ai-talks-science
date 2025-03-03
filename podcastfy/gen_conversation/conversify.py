import os
import re
from typing import Dict, Any, Optional, List
from podcastfy.content_parser.content_extractor import ContentExtractor
from podcastfy.utils.config import Config
from podcastfy.utils.config_conversation import ConversationConfig
from podcastfy.content_generator import LLMBackend
from podcastfy.template_reader import TemplateLoader
from podcastfy.gen_conversation.introduction import extract_info, generate_introduction
from podcastfy.gen_conversation.qa_session import generate_qainformation, generate_qasession
from podcastfy.gen_conversation.close_podcast import close_podcast
import logging
logger = logging.getLogger(__name__)

class ContentCleanerMixin:
    """
    Mixin class containing common transcript cleaning operations.
    
    Provides reusable cleaning methods that can be used by different content generation strategies.
    Methods use protected naming convention (_method_name) as they are intended for internal use
    by the strategies.
    """
    
    @staticmethod
    def _clean_scratchpad(text: str) -> str:
        """
        Remove scratchpad blocks, plaintext blocks, standalone triple backticks, any string enclosed in brackets, and underscores around words.
        """
        try:
            import re
            pattern = r'```scratchpad\n.*?```\n?|```plaintext\n.*?```\n?|```\n?|```xml\n?|\[.*?\]'
            cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
            # Remove "xml" if followed by </Person1> or </Person2>
            cleaned_text = re.sub(r"xml(?=\s*</Person[12]>)", "", cleaned_text)
            # Remove underscores around words
            cleaned_text = re.sub(r'_(.*?)_', r'\1', cleaned_text)
            return cleaned_text.strip()
        except Exception as e:
            logger.error(f"Error cleaning scratchpad content: {str(e)}")
            return text

    @staticmethod
    def _clean_tss_markup(
        input_text: str, 
        additional_tags: List[str] = ["Person1", "Person2"]
    ) -> str:
        """
        Remove unsupported TSS markup tags while preserving supported ones.
        """
        try:
            input_text = ContentCleanerMixin._clean_scratchpad(input_text)
            supported_tags = ["speak", "lang", "p", "phoneme", "s", "sub"]
            supported_tags.extend(additional_tags)

            pattern = r"</?(?!(?:" + "|".join(supported_tags) + r")\b)[^>]+>"
            cleaned_text = re.sub(pattern, "", input_text)
            cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
            cleaned_text = re.sub(r"\*", "", cleaned_text)

            for tag in additional_tags:
                cleaned_text = re.sub(
                    f'<{tag}>(.*?)(?=<(?:{"|".join(additional_tags)})>|$)',
                    f"<{tag}>\\1</{tag}>",
                    cleaned_text,
                    flags=re.DOTALL,
                )
            


            return cleaned_text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning TSS markup: {str(e)}")
            return input_text

    @staticmethod
    def _fix_alternating_tags(transcript: str) -> str:
        """
        Ensures transcript has properly alternating Person1 and Person2 tags.
        
        Merges consecutive same-person tags and ensures proper tag alternation
        throughout the transcript.
        
        Args:
            transcript (str): Input transcript text that may have consecutive same-person tags
            
        Returns:
            str: Transcript with properly alternating tags and merged content
            
        Example:
            Input:
                <Person1>Hello</Person1>
                <Person1>World</Person1>
                <Person2>Hi</Person2>
            Output:
                <Person1>Hello World</Person1>
                <Person2>Hi</Person2>
                
        Note:
            Returns original transcript if cleaning fails
        """
        try:
            # Split into individual tag blocks while preserving tags
            pattern = r'(<Person[12]>.*?</Person[12]>)'
            blocks = re.split(pattern, transcript, flags=re.DOTALL)
            
            # Filter out empty/whitespace blocks
            blocks = [b.strip() for b in blocks if b.strip()]
            
            merged_blocks = []
            current_content = []
            current_person = None
            
            for block in blocks:
                # Extract person number and content
                match = re.match(r'<Person([12])>(.*?)</Person\1>', block, re.DOTALL)
                if not match:
                    continue
                    
                person_num, content = match.groups()
                content = content.strip()
                
                if current_person == person_num:
                    # Same person - append content
                    current_content.append(content)
                else:
                    # Different person - flush current content if any
                    if current_content:
                        merged_text = " ".join(current_content)
                        merged_blocks.append(f"<Person{current_person}>{merged_text}</Person{current_person}>")
                    # Start new person
                    current_person = person_num
                    current_content = [content]
            
            # Flush final content
            if current_content:
                merged_text = " ".join(current_content)
                merged_blocks.append(f"<Person{current_person}>{merged_text}</Person{current_person}>")
                
            return "\n".join(merged_blocks)
            
        except Exception as e:
            logger.error(f"Error fixing alternating tags: {str(e)}")
            return transcript  # Return original if fixing fails


class PodcastGenerator:
    """
    A class that orchestrates the generation of podcast content from academic papers.
    
    This class manages the workflow of extracting paper information, generating an
    introduction, Q&A session, and closing segment for a podcast based on academic papers.
    It maintains state to ensure proper execution order and prevents redundant operations.
    """
    
    def __init__(self, 
                 pdf_path: str, 
                 config: Dict[str, Any], 
                 llm: LLMBackend,
                 template_reader: Optional[TemplateLoader] = None):
        """
        Initialize the podcast generator with necessary resources.
        
        Args:
            pdf_path: Path to the PDF file to process
            config: Configuration dictionary for conversation style and parameters
            llm: LLMBackend instance for generation tasks
            template_reader: Optional TemplateLoader instance, created if not provided
        """
        self.pdf_path = pdf_path
        self.config = config
        self.llm = llm
        self.template_reader = template_reader or TemplateLoader()
        
        # Initialize extractors
        self.content_extractor = ContentExtractor()
        
        # Load templates
        common_instructions=self.template_reader.get_template("podcastfy/configs/common_instructions.md")
        self.config['common_instructions'] = self.template_reader.fill_template(common_instructions, self.config)
        
        introduction=self.template_reader.get_template("./podcastfy/configs/instructions_introduction.md")
        closing=self.template_reader.get_template("./podcastfy/configs/instructions_end.md")
        qa_session=self.template_reader.get_template("./podcastfy/configs/generate_qa_session.md")
        self.templates = {
            "introduction": self.template_reader.fill_template(introduction, self.config),
            "qa": self.template_reader.get_template("./podcastfy/configs/generate_qa.md"),
            "qa_session": self.template_reader.fill_template(qa_session, self.config),
            "closing": self.template_reader.fill_template(closing, self.config)
        }

        # Extract PDF content once
        self._context = None
        
        # State tracking
        self._paper_info = None
    
    @property
    def pdf_context(self) -> str:
        """
        Extract and cache the PDF content.
        
        Returns:
            String containing the extracted PDF content
        """
        if self._context is None:
            self._context = self.content_extractor.extract_content(self.pdf_path)
        return self._context
    
    @property
    def paper_info(self) -> Dict[str, Any]:
        """
        Extract and cache paper information from the PDF.
        
        Returns:
            Dictionary with extracted paper information
        """
        if self._paper_info is None:
            self._paper_info = extract_info(self.pdf_path, self.llm)
        return self._paper_info
    
    def stitch_podcast(self, segments: Dict[str, str]) -> str:
        """
        Combine podcast segments into a complete transcript.
        
        Args:
            segments: Dictionary with introduction, qa_session, and closing segments
            
        Returns:
            Complete podcast transcript as a string
        """
        sections = [
            segments["introduction"],
            segments["qa_session"],
            segments["closing"]
        ]
        
        return "\n".join(sections)

    @staticmethod
    def clean_transcript(transcript: str) -> str:
        """
        Apply all cleaning steps to prepare transcript for TTS engine.
        
        Args:
            transcript: Raw transcript text from LLM
                
        Returns:
            Cleaned transcript ready for TTS processing
        """

        cleaned = ContentCleanerMixin._clean_scratchpad(transcript)
        cleaned = ContentCleanerMixin._clean_tss_markup(cleaned)
        cleaned = ContentCleanerMixin._fix_alternating_tags(cleaned)
        return cleaned
    
    def gen_interview_transcript(self, clean_transcript: bool = True) -> Dict[str, str] | str:
        """
        Orchestrate the full podcast generation process.
        
        This method executes all steps in the correct order:
        1. Extract paper information
        2. Generate introduction
        3. Generate Q&A session
        4. Generate closing segment
        
        Args:
            return_segments: If True, returns a dictionary with separate segments
                            If False, returns a single string with all segments stitched together
            
        Returns:
            Complete podcast transcript or dictionary with segments based on return_segments
        """
        introduction = generate_introduction(
            self.paper_info, 
            self.templates["introduction"], 
            self.config, 
            self.llm
        )
        qa_info=generate_qainformation(
            self.pdf_context, 
            self.template_reader, 
            self.llm
        )
        main_section=generate_qasession(qa_info, 
                                        self.templates['qa_session'], 
                                        self.config,
                                        self.llm)
        closing_segment=close_podcast(self.config,
                                      self.templates["closing"],
                                      self.llm,
                                      self.paper_info)
        
        combined_segments=self.stitch_podcast({
            "introduction": introduction,
            "qa_session": main_section,
            "closing": closing_segment
        })

        print("Transcripts before cleaning")
        print(combined_segments)
        print()
        if clean_transcript:
            combined_segments=self.clean_transcript(combined_segments)
        print("Transcripts after cleaning")
        print(combined_segments)
        return combined_segments

if __name__ == "__main__":
    # Test template loading functionality
    PROJECT_PATH = "./projects/project_1"
    INPUTS = {
        "pdf_path": os.path.join(PROJECT_PATH, "2412.18925v1.pdf"),
        "base_config": "./podcastfy/configs/config.yaml",
        "conversation_config": os.path.join(PROJECT_PATH, "conversation_config.yaml")
    }

    # Load configs
    config = Config(INPUTS["base_config"])#.to_dict()
    conversation_config = ConversationConfig(path=INPUTS["conversation_config"]).to_dict()
    #llm=None # ignore for the moment, test only template reading
    model_name = "gemini-1.5-flash-latest"
    api_key_label = "GEMINI_API_KEY"

    llm = LLMBackend(
        is_local=False, 
        temperature=0.7, 
        max_output_tokens=2048, 
        model_name=model_name, 
        api_key_label=api_key_label
    )
    pgenerator=PodcastGenerator(INPUTS["pdf_path"], conversation_config, llm)

    # Test that templates are loaded correctly
    # for template_name in pgenerator.templates.keys():
    #     print("")
    #     if template_name == "qa_session":
    #         print(f"Template {template_name} loaded, content: {pgenerator.templates[template_name]}...")
    #     
    pgenerator.gen_interview_transcript()