"""
PDF Extractor Module

This module provides functionality to extract text content from PDF files.
It handles the reading of PDF files, text extraction, and normalization of
the extracted content, including handling of special characters and accents.
"""

import pymupdf
import logging
import os
import unicodedata
import re

logger = logging.getLogger(__name__)

class PDFExtractor:
	def extract_content(self, file_path: str, remove_refs: bool = True) -> str:
		"""
		Extract text content from a PDF file, handling foreign characters and special characters.
		Accents are removed from the text.

		Args:
			file_path (str): Path to the PDF file.

		Returns:
			str: Extracted text content with accents removed and properly handled characters.
		"""
		try:
			doc = pymupdf.open(file_path)
			content = " ".join(page.get_text() for page in doc)
			doc.close()
			
			# Normalize the text to handle special characters and remove accents
			normalized_content = unicodedata.normalize('NFKD', content)
			if remove_refs:
				# Remove references (e.g., [1], [2], etc.)
				normalized_content = self._ref_cleaner(normalized_content)

			return normalized_content
		except Exception as e:
			logger.error(f"Error extracting PDF content: {str(e)}")
			raise
	
	def extract_page1(self, file_path: str) -> str:
		"""
		Extract text content from the first page of a PDF file.
		
		Args:
			file_path (str): Path to the PDF file.
			remove_refs (bool): Whether to remove references from the content.
			
		Returns:
			str: Extracted text content from the first page.
		"""
		try:
			doc = pymupdf.open(file_path)
			content = doc.load_page(0).get_text()
			doc.close()

			return content
		except Exception as e:
			logger.error(f"Error extracting first page content: {str(e)}")
			raise
	def _ref_cleaner(self, text: str) -> str:
		"""
		Remove references and bibliography sections from the text.
		Only searches for references in the lower half of the document.
		The aim is to reduce the context for the LLM.
		
		Args:
			text (str): Input text to clean
			
		Returns:
			str: Text with references section removed
		"""
		# Common section headers for references:
			# (?:^|\n)    = Match either start of text (^) or newline, non-capturing group (?:)
			# \s*         = Zero or more whitespace characters
			# References? = Match "Reference" or "References" (? makes 's' optional)
			# \b          = Word boundary (ensures whole word match)
			# [:\s\n]     = Character class matching colon, whitespace or newline
		ref_patterns = [
			r'(?:^|\n)\s*References?\b[:\s\n]',
			r'(?:^|\n)\s*Bibliography\b[:\s\n]',
			r'(?:^|\n)\s*Literature cited\b[:\s\n]',
			r'(?:^|\n)\s*Works cited\b[:\s\n]'
		]
		
		# For safert: only look in the lower half of the document
		half_length = len(text) // 2
		lower_half = text[half_length:]
		
		# Find the first occurrence of any reference section in the lower half
		min_pos = len(lower_half)
		for pattern in ref_patterns:
			match = re.search(pattern, lower_half, re.IGNORECASE)
			if match and match.start() < min_pos:
				min_pos = match.start()
		
		# If reference section found in lower half, return text up to that point
		if min_pos < len(lower_half):
			return text[:half_length + min_pos].strip()
		
		# If no reference section found in lower half, return the full text
		return text

	def extract_figure(self, file_path: str, figure_number: int = 0) -> str:
		"""
		Extract a specific figure from a PDF file and save it as JPEG.
		
		Args:
			file_path (str): Path to the PDF file.
			figure_number (int): Index of the figure to extract (0-based).
			
		Returns:
			str: Path to the saved figure image.
		"""
		try:
			doc = pymupdf.open(file_path)
			output_dir = os.path.dirname(file_path)
			output_path = os.path.join(output_dir, f"jpeg_{figure_number}.jpg")
			
			image_count = 0
			for page_num in range(len(doc)):
				page = doc.load_page(page_num)
				image_list = page.get_images(full=True)
				
				for img_index, img in enumerate(image_list):
					if image_count == figure_number:
						xref = img[0]
						base_image = doc.extract_image(xref)
						image_bytes = base_image["image"]
						
						with open(output_path, "wb") as f:
							f.write(image_bytes)
						
						logger.info(f"Saved figure {figure_number} to {output_path}")
						return output_path
					
					image_count += 1
			
			logger.warning(f"Figure {figure_number} not found in document")
			return ""
		except Exception as e:
			logger.error(f"Error extracting figure: {str(e)}")
			raise

	def extract_all_figures(self, file_path: str) -> dict:
		"""
		Extract all figures from a PDF file and save them as JPEGs.
		
		Args:
			file_path (str): Path to the PDF file.
			
		Returns:
			dict: Dictionary mapping figure numbers to file paths.
		"""
		try:
			doc = pymupdf.open(file_path)
			output_dir = os.path.dirname(file_path)
			figures = {}
			
			image_count = 0
			for page_num in range(len(doc)):
				page = doc.load_page(page_num)
				image_list = page.get_images(full=True)
				
				for img_index, img in enumerate(image_list):
					xref = img[0]
					base_image = doc.extract_image(xref)
					image_bytes = base_image["image"]
					
					output_path = os.path.join(output_dir, f"jpeg_{image_count}.jpg")
					with open(output_path, "wb") as f:
						f.write(image_bytes)
					
					figures[image_count] = output_path
					image_count += 1
			
			logger.info(f"Extracted {image_count} figures from {file_path}")
			return figures
		except Exception as e:
			logger.error(f"Error extracting figures: {str(e)}")
			raise


def main(test: str, seed: int = 42) -> None:
	"""
	Test the PDFExtractor class with a specific PDF file.

	Args:
		seed (int): Random seed for reproducibility. Defaults to 42.
	"""
	if test == 'content':
		# Set the random seed
		import random
		random.seed(seed)

		# Get the absolute path of the script
		script_dir = os.path.dirname(os.path.abspath(__file__))
		
		# Construct the path to the PDF file
		pdf_path = os.path.join(script_dir, '..', '..', 'tests', 'data', 'file.pdf')
		
		extractor = PDFExtractor()

		try:
			content = extractor.extract_content(pdf_path)
			print("PDF content extracted successfully:")
			print(content[:500] + "..." if len(content) > 500 else content)
		except Exception as e:
			print(f"An error occurred: {str(e)}")

	elif test == 'figures':
		extractor = PDFExtractor()
		
		# Path to the specified PDF file
		pdf_path = "./projects/project_6/nature_salmonella.pdf"
		
		# Make sure the file exists
		if not os.path.exists(pdf_path):
			print(f"File not found: {pdf_path}")
			return
		
		# Test single figure extraction
		print("Extracting first figure...")
		figure_path = extractor.extract_figure(pdf_path, 0)
		if figure_path:
			print(f"First figure saved to: {figure_path}")
		else:
			print("Failed to extract first figure")
		
		# Test all figures extraction
		print("\nExtracting all figures...")
		figures = extractor.extract_all_figures(pdf_path)
		print(f"Extracted {len(figures)} figures:")
		for fig_num, path in figures.items():
			print(f"Figure {fig_num}: {path}")
		
	else:
		return True



if __name__ == "__main__":
	main()