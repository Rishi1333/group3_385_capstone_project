"""
Vision Processor Service

Handles medical image uploads and analysis using multimodal LLM.
"""

import base64
import io
import logging
from typing import Dict, Any, Optional
from PIL import Image
import ollama

# Configure logging
logger = logging.getLogger(__name__)


class VisionProcessor:
    """
    Processes medical images using multimodal LLM.
    """
    
    def __init__(self, llm_model: str = "gemini-3-flash-preview"):
        """
        Initialize the vision processor.
        
        Args:
            llm_model: Multimodal LLM model
        """
        self.llm_model = llm_model
    
    def encode_image(self, image_data: bytes) -> str:
        """
        Encode image to base64.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Base64 encoded string
        """
        return base64.b64encode(image_data).decode('utf-8')
    
    def process_image(
        self,
        image_data: bytes,
        context: str = None
    ) -> Dict[str, Any]:
        """
        Process a medical image.
        
        Args:
            image_data: Raw image bytes
            context: Optional context about what to look for
            
        Returns:
            Analysis results
        """
        # Encode image
        encoded = self.encode_image(image_data)
        
        # Build prompt
        prompt = self._build_image_prompt(context)
        
        try:
            # Send to multimodal model
            response = ollama.chat(
                model=self.llm_model,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [encoded]
                }]
            )
            
            analysis = response["message"]["content"]
            
            return {
                "success": True,
                "analysis": analysis,
                "model": self.llm_model
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_image_file(self, filepath: str, context: str = None) -> Dict[str, Any]:
        """
        Process an image from file path.
        
        Args:
            filepath: Path to image file
            context: Optional context
            
        Returns:
            Analysis results
        """
        try:
            with open(filepath, 'rb') as f:
                image_data = f.read()
            return self.process_image(image_data, context)
        except Exception as e:
            logger.error(f"Error reading image file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_image_prompt(self, context: str = None) -> str:
        """Build prompt for image analysis."""
        base_prompt = """
You are a medical image analysis assistant. Analyze the provided medical image and provide detailed observations.

IMPORTANT:
- Describe what you see in clinical terms
- Note any abnormal findings
- Do NOT provide diagnosis - leave that to medical professionals
- Include relevant observations about the image quality and any artifacts

"""
        
        if context:
            base_prompt += f"\nSpecific context: {context}\n"
        
        base_prompt += """
Provide your analysis in a structured format:
1. Image Type/Modality
2. Description of findings
3. Notable observations
4. Limitations or caveats
"""
        
        return base_prompt
    
    def validate_image(self, image_data: bytes) -> Dict[str, Any]:
        """
        Validate that the image is valid and processable.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Validation result
        """
        try:
            # Try to open with PIL
            img = Image.open(io.BytesIO(image_data))
            
            # Check format
            valid_formats = ['JPEG', 'PNG', 'JPG', 'WEBP', 'BMP']
            if img.format not in valid_formats:
                return {
                    "valid": False,
                    "error": f"Unsupported image format: {img.format}"
                }
            
            # Check size (max 10MB)
            if len(image_data) > 10 * 1024 * 1024:
                return {
                    "valid": False,
                    "error": "Image too large (max 10MB)"
                }
            
            # Check dimensions
            width, height = img.size
            if width < 32 or height < 32:
                return {
                    "valid": False,
                    "error": "Image too small (min 32x32)"
                }
            
            return {
                "valid": True,
                "format": img.format,
                "size": img.size,
                "mode": img.mode
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Invalid image: {str(e)}"
            }


# Singleton instance
_vision_processor = None

def get_vision_processor() -> VisionProcessor:
    """Get or create the singleton VisionProcessor instance."""
    global _vision_processor
    if _vision_processor is None:
        _vision_processor = VisionProcessor()
    return _vision_processor
