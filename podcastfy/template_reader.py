import os
import logging
from typing import Dict, Optional

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


if __name__ == "__main__":
    loader = TemplateLoader()
    template = loader.get_template("./podcastfy/configs/longform_template.md")
    print(template)