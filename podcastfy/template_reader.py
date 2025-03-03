import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class TemplateLoader:
    def __init__(self):
        self.template_cache: Dict[str, str] = {}
    def load_content(self, template_path: str) -> str:
        """
        Load template from file with path validation.

        Args:
            template_path: Direct path to template file

        Returns:
            Formatted template string
        """
        basename = os.path.splitext(os.path.basename(template_path))[0]
        
        if basename in self.template_cache:
            return self.template_cache[basename]

        if not os.path.exists(template_path):
            logger.warning(f"Template file not found: {template_path}")
            return ""

        if not template_path.endswith(('.md', '.txt')):
            logger.warning(f"Invalid template file format. Use .md or .txt: {template_path}")
            return ""

        try:
            with open(template_path, 'r') as f:
                template = f.read()
            self.template_cache[basename] = template
            return template
        except Exception as e:
            logger.error(f"Error loading template {template_path}: {str(e)}")
            return ""
    def fill_template(self, template: str, data: dict) -> str:
        """
        Format template with provided data.
        
        Args:
            template: Template string with {placeholders}
            data: Dict of values to insert
            
        Returns:
            Formatted string with replacements
        """
        try:
            self.template = template.format(**data)
            return self.template
        except KeyError as e:
            logger.warning(f"Missing template variable: {str(e)}")
            return template
        except Exception as e:
            logger.error(f"Error formatting template: {str(e)}")
            return template
    def get_template(self, template_type: str) -> str:
        """
        Load and format template for LLM consumption.
        
        Args:
            template_type: Template file path
            
        Returns:
            Clean, formatted template string
        """
        template = self.load_content(template_type)
        

        # Format for LLM consumption
        formatted = template.replace('\n\n', '\n')
        formatted = ' '.join(formatted.split())
        formatted = formatted.strip()
        
        # Add clear section markers
        #if formatted:
        #    formatted = f"### Instructions ###\n{formatted}\n### End Instructions ###"
            
        return formatted

    def get_sections(self, template_name: str, section_markers: List[str], remove_hashtags: bool = False) -> str:
        """
        Extract specific sections from the template based on section markers.
        
        Args:
            template_path: Path to the template file
            section_markers: List of section markers to find (e.g., ["Background", "MainContributions"])
            remove_hashtags: Whether to remove hashtags from section titles

        Returns:
            The content of the specified sections
        """
        full_template=self.template_cache[template_name]
        if not full_template:
            return ""
        
        sections_content = []
        current_section = None
        capture = False
        section_text = ""
        
        for line in full_template.splitlines():
            if line.startswith("#"):
                # If we were capturing a section, save it before moving to new section
                if current_section and capture:
                    sections_content.append(current_section + section_text)
                
                # Start a new section
                section_name = line.strip("# ").strip()
                if remove_hashtags:
                    current_section = section_name + "\n"
                else:
                    current_section = line + "\n"
                    
                section_text = ""
                
                # Check if any section marker is in the section name
                capture = any(marker in section_name for marker in section_markers)
            elif capture:
                section_text += line + "\n"

        
        # Add the last section if it was being captured
        if current_section and capture:
            sections_content.append(current_section + section_text)
        
        combined_content = "\n".join(sections_content).strip()
        return combined_content


if __name__ == "__main__":
    # Basic test for template loading
    loader = TemplateLoader()
    template = loader.get_template("./podcastfy/configs/generate_qa.md")
    print("Full template loaded:", bool(template))
    
    # Test the get_sections method
    print("\nTesting get_sections method:")
    sections = ["Background"]
    
    section_content = loader.get_sections("generate_qa", ["1. Background", "SPECIFICITYSTRATEGY:", "TASK:", "REQUIREMENTS:"], remove_hashtags=True)
    section_found = bool(section_content)    
    print(section_content)