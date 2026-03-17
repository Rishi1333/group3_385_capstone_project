"""
Dynamic Triage Agent

Implements a dynamic, LLM-driven state machine for patient intake flow.
Prioritizes chief complaint analysis and adapts conversation based on context.

States:
- INITIAL_INPUT: Waiting for initial patient input
- RAPID_TRIAGE: Analyzing urgency of chief complaint
- EMERGENCY_PATH: Handling critical/emergency situations
- PHI_COLLECTION: Collecting essential patient health information
- DEEP_DIVE: Gathering detailed symptom information
- IMAGE_INTAKE: Requesting and processing medical images
- CLINICAL_SUMMARY: Generating pre-consultation summary
- COMPLETE: Session finished
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

try:
    import ollama
except ImportError:
    ollama = None

# Configure logging
logger = logging.getLogger(__name__)


class TriageState(Enum):
    """States for the dynamic triage flow."""
    INITIAL_INPUT = "INITIAL_INPUT"
    RAPID_TRIAGE = "RAPID_TRIAGE"
    EMERGENCY_PATH = "EMERGENCY_PATH"
    PHI_COLLECTION = "PHI_COLLECTION"
    DEEP_DIVE = "DEEP_DIVE"
    IMAGE_INTAKE = "IMAGE_INTAKE"
    CLINICAL_SUMMARY = "CLINICAL_SUMMARY"
    COMPLETE = "COMPLETE"


class UrgencyLevel(Enum):
    """Urgency classification levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"


class DynamicTriageAgent:
    """
    Dynamic, LLM-driven triage agent.
    
    Adapts conversation flow based on patient input rather than
    following a rigid script. Prioritizes safety and contextual
    understanding.
    """
    
    # Maximum questions to ask in deep dive phase
    MAX_QUESTIONS = 10
    
    # Minimum PHI required before proceeding
    REQUIRED_PHI = ['age', 'sex']
    
    # Visual symptom indicators for image intake
    VISUAL_INDICATORS = [
        'rash', 'swelling', 'wound', 'bruise', 'lesion', 'cut',
        'discoloration', 'growth', 'mark', 'spot', 'lump', 'burn',
        'blister', 'sore', 'infection', 'abscess'
    ]
    
    def __init__(
        self,
        session_id: str,
        llm_model: str = "gemini-3-flash-preview",
        rag_service=None,
        vision_processor=None
    ):
        """
        Initialize the dynamic triage agent.
        
        Args:
            session_id: Unique session identifier
            llm_model: LLM model for conversation
            rag_service: RAG service for medical knowledge retrieval
            vision_processor: Vision processor for image analysis
        """
        self.session_id = session_id
        self.llm_model = llm_model
        self.rag_service = rag_service
        self.vision_processor = vision_processor
        
        # Current state
        self.state = TriageState.INITIAL_INPUT
        
        # Clinical data storage
        self.clinical_data = {
            "chief_complaint": "",
            "symptoms": [],
            "symptom_details": {},  # Detailed info per symptom
            "medical_history": [],
            "medications": [],
            "allergies": [],
            "risk_factors": [],
            "family_history": [],
            "lifestyle": {
                "smoking": None,
                "alcohol": None,
                "exercise": None
            },
            "patient_profile": {
                "age": None,
                "sex": None
            },
            "image_findings": None,
            "urgency_level": None
        }
        
        # Conversation history
        self.message_history = []
        
        # Tracking
        self.questions_asked = 0
        self.current_question_type = None
        self.pending_image_request = False
        self.image_processed = False  # Track if an image has been processed
        self._previous_state = None  # Track previous state before image intake
        
        # Add system message
        self._add_message("system", self._get_system_prompt())
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the triage agent."""
        return """You are a medical triage assistant for a virtual clinic. Your role is to:

1. Gather patient symptoms and medical history through natural conversation
2. Be empathetic, clear, and professional
3. Ask one question at a time to avoid overwhelming the patient
4. Collect necessary clinical information for assessment
5. NEVER provide diagnosis - only gather information
6. Always include appropriate medical disclaimers

IMPORTANT GUIDELINES:
- Start by understanding the chief complaint before collecting demographic info
- If the patient mentions severe symptoms (chest pain, difficulty breathing, stroke symptoms), 
  immediately assess urgency and provide appropriate guidance
- Ask about symptom duration, severity, and progression
- Adapt your questions based on the patient's responses
- Be conversational and supportive

Do not mention that you are an AI. Use natural, empathetic language."""
    
    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def process_input(self, user_input: str, image_data: bytes = None) -> Dict[str, Any]:
        """
        Process user input based on current state.
        
        Args:
            user_input: User's text input
            image_data: Optional image data
            
        Returns:
            Response dict with next action and message
        """
        # Add user message to history
        self._add_message("user", user_input)
        
        # Handle image if provided
        if image_data and self.state == TriageState.IMAGE_INTAKE:
            return self._process_image(image_data)
        
        # Process based on current state
        state_handlers = {
            TriageState.INITIAL_INPUT: self._handle_initial_input,
            TriageState.RAPID_TRIAGE: self._handle_rapid_triage,
            TriageState.EMERGENCY_PATH: self._handle_emergency_path,
            TriageState.PHI_COLLECTION: self._handle_phi_collection,
            TriageState.DEEP_DIVE: self._handle_deep_dive,
            TriageState.IMAGE_INTAKE: self._handle_image_intake,
            TriageState.CLINICAL_SUMMARY: self._handle_clinical_summary,
        }
        
        handler = state_handlers.get(self.state)
        if handler:
            return handler(user_input)
        
        return self._generate_response(
            "I apologize, but I'm not sure how to proceed. Let me help you get the assistance you need.",
            state=self.state
        )
    
    def _handle_initial_input(self, user_input: str) -> Dict[str, Any]:
        """Handle initial patient input - extract chief complaint and assess urgency."""
        self.clinical_data["chief_complaint"] = user_input
        
        # Extract any immediately available information
        self._extract_phi_from_text(user_input)
        self._extract_symptoms_from_text(user_input)
        
        # Transition to rapid triage
        self.state = TriageState.RAPID_TRIAGE
        return self._handle_rapid_triage(user_input)
    
    def _handle_rapid_triage(self, user_input: str) -> Dict[str, Any]:
        """Assess urgency of the chief complaint using LLM reasoning."""
        
        # Use LLM to assess urgency
        urgency_assessment = self._assess_urgency(user_input)
        self.clinical_data["urgency_level"] = urgency_assessment["urgency"]
        
        if urgency_assessment["urgency"] == UrgencyLevel.CRITICAL.value:
            self.state = TriageState.EMERGENCY_PATH
            return self._generate_response(
                urgency_assessment.get("emergency_message", 
                    "Based on what you've described, this may require immediate medical attention."),
                state=TriageState.EMERGENCY_PATH,
                urgency=UrgencyLevel.CRITICAL.value,
                action="EMERGENCY_SERVICES"
            )
        
        # Not critical - proceed to PHI collection
        self.state = TriageState.PHI_COLLECTION
        
        # Generate appropriate response
        if urgency_assessment["urgency"] == UrgencyLevel.HIGH.value:
            response = urgency_assessment.get("initial_response", 
                "I understand you're experiencing something concerning. Let me gather some information to help you better.")
        else:
            response = "Thank you for sharing that. Let me ask a few questions to better understand your situation."
        
        # Add PHI question if we don't have it
        missing_phi = self._get_missing_phi()
        if missing_phi:
            phi_question = self._generate_phi_question(missing_phi)
            response += f" {phi_question}"
        
        return self._generate_response(response, state=TriageState.PHI_COLLECTION)
    
    def _handle_emergency_path(self, user_input: str) -> Dict[str, Any]:
        """Handle emergency situation - provide guidance and end session."""
        
        emergency_response = """
🚨 **IMMEDIATE ASSISTANCE NEEDED** 🚨

Based on what you've described, you should seek immediate medical attention.

**Please contact:**
- 📞 Emergency Services: **911** (US) / **999** (UK) / **112** (EU)
- 📞 National Suicide Prevention: **988** (US)

**While waiting for help:**
- Stay calm and try to rest
- Do not drive yourself
- If possible, have someone stay with you
- Follow any specific guidance from emergency services

⚠️ **DISCLAIMER**: I am an AI assistant, not a doctor. This is not medical advice. 
Please seek immediate professional medical help for any life-threatening condition.
"""
        
        self.state = TriageState.COMPLETE
        return self._generate_response(
            emergency_response,
            state=TriageState.COMPLETE,
            urgency=UrgencyLevel.CRITICAL.value,
            action="EMERGENCY_SERVICES",
            end_session=True
        )
    
    def _handle_phi_collection(self, user_input: str) -> Dict[str, Any]:
        """Collect essential patient health information."""
        
        # Extract PHI from response
        self._extract_phi_from_text(user_input)
        
        # Check if we have minimum PHI
        missing_phi = self._get_missing_phi()
        
        if missing_phi:
            # Still need more PHI
            phi_question = self._generate_phi_question(missing_phi)
            return self._generate_response(phi_question, state=TriageState.PHI_COLLECTION)
        
        # Have minimum PHI - transition to deep dive
        self.state = TriageState.DEEP_DIVE
        return self._handle_deep_dive(user_input)
    
    def _handle_deep_dive(self, user_input: str) -> Dict[str, Any]:
        """Gather detailed symptom information dynamically."""
        
        # Extract clinical information from response
        self._extract_clinical_info(user_input)
        
        # Check if we have sufficient info
        if self._has_sufficient_info() or self.questions_asked >= self.MAX_QUESTIONS:
            self.state = TriageState.CLINICAL_SUMMARY
            return self._handle_clinical_summary(user_input)
        
        # Generate next contextual question
        next_question = self._generate_contextual_question()
        self.questions_asked += 1
        
        return self._generate_response(next_question, state=TriageState.DEEP_DIVE)
    
    def _handle_image_intake(self, user_input: str) -> Dict[str, Any]:
        """Handle image intake state - either process image or continue conversation."""
        
        # Check if user declined to provide image
        if any(phrase in user_input.lower() for phrase in ['no image', 'no photo', 'no picture', 'skip', 'not now', 'later', 'no', 'dont have']):
            self.pending_image_request = False
            self.image_processed = True  # Mark as processed so we don't ask again
            self.state = TriageState.DEEP_DIVE
            return self._handle_deep_dive("No image provided, continuing with questions.")
        
        # Ask again for image or continue
        if 'yes' in user_input.lower() or 'sure' in user_input.lower() or 'okay' in user_input.lower():
            return self._generate_response(
                "Please upload the image when you're ready. You can also describe what you're seeing if that's easier.",
                state=TriageState.IMAGE_INTAKE,
                request_image=True
            )
        
        # Continue with text conversation - user provided text instead of image
        self.pending_image_request = False
        self.image_processed = True  # Mark as processed so we don't ask again
        self.state = TriageState.DEEP_DIVE
        return self._handle_deep_dive(user_input)
    
    def _handle_clinical_summary(self, user_input: str) -> Dict[str, Any]:
        """Generate clinical summary and end session."""
        
        # Check if user has documents to share before generating report
        if not self.pending_image_request and not self.image_processed:
            self.pending_image_request = True
            return self._generate_response(
                "Before I generate your clinical summary, do you have any documents or images "
                "related to your symptoms that you'd like to share? This could include photos of "
                "a rash, swelling, or any visible symptoms, or documents such as previous test results.",
                state=TriageState.CLINICAL_SUMMARY,
                request_image=True,
                end_session=False
            )
        
        # User indicated they have documents - transition to image intake
        if self.pending_image_request and not self.image_processed:
            # Check if user said no/don't have
            if any(phrase in user_input.lower() for phrase in ['no image', 'no photo', 'no picture', 'skip', 'not now', 'later', 'no', 'dont have', 'none', "don't have", "do not have"]):
                self.pending_image_request = False
                self.image_processed = True
            else:
                self._previous_state = TriageState.CLINICAL_SUMMARY
                self.state = TriageState.IMAGE_INTAKE
                return self._generate_response(
                    "Please upload your document or image now. You can drag and drop or click to select a file.",
                    state=TriageState.IMAGE_INTAKE,
                    request_image=True
                )
        
        # Generate the clinical summary
        summary = self._generate_clinical_summary()
        self.state = TriageState.COMPLETE
        
        return self._generate_response(
            summary,
            state=TriageState.COMPLETE,
            end_session=True,
            summary=self.clinical_data
        )
    
    def _process_image(self, image_data: bytes) -> Dict[str, Any]:
        """Process uploaded image using vision processor."""
        
        # Mark image as processed regardless of outcome
        self.image_processed = True
        self.pending_image_request = False
        
        # Determine next state: return to previous state if set, otherwise DEEP_DIVE
        next_state = self._previous_state if self._previous_state else TriageState.DEEP_DIVE
        self._previous_state = None  # Clear after use
        
        if not self.vision_processor:
            self.state = next_state
            return self._generate_response(
                "I received your image, but I'm currently unable to process images. "
                "Let me continue with some questions about what you're experiencing.",
                state=next_state
            )
        
        try:
            # Analyze image
            context = f"Chief complaint: {self.clinical_data['chief_complaint']}"
            result = self.vision_processor.process_image(image_data, context)
            
            if result.get("success"):
                self.clinical_data["image_findings"] = result.get("analysis", "")
                
                # Generate follow-up questions based on image
                follow_up = self._generate_image_followup(result.get("analysis", ""))
                
                self.state = next_state
                
                return self._generate_response(
                    f"I've reviewed the image. {follow_up}",
                    state=next_state,
                    image_analysis=result.get("analysis", "")
                )
            else:
                self.state = next_state
                return self._generate_response(
                    "I had trouble analyzing the image. Could you describe what you're seeing in the affected area?",
                    state=next_state
                )
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self.state = next_state
            return self._generate_response(
                "I encountered an issue with the image. Let's continue with some questions.",
                state=next_state
            )
    
    def _assess_urgency(self, user_input: str) -> Dict[str, Any]:
        """Assess urgency using LLM reasoning."""
        
        prompt = f"""You are a medical safety classifier. Analyze this patient complaint for urgency level.

PATIENT INPUT: "{user_input}"

URGENCY CLASSIFICATION RULES:

1. CRITICAL - Life-threatening, requires immediate emergency services (911)
   Examples: chest pain with shortness of breath, stroke symptoms, severe bleeding, 
   suicidal ideation with intent, inability to breathe, severe allergic reaction

2. HIGH - Urgent medical situation requiring care within hours
   Examples: high fever with confusion, severe pain, moderate bleeding, suspected fracture

3. MODERATE - Should see a doctor within 24-48 hours
   Examples: persistent symptoms, worsening condition, new concerning symptoms

4. LOW - Can be addressed with routine appointment or self-care
   Examples: mild symptoms, general questions, follow-up inquiries

IMPORTANT CONTEXT CONSIDERATIONS:
- "Chest pain after eating spicy food" = likely GERD, not emergency
- "Chest pain with arm numbness and sweating" = likely cardiac emergency
- "Headache after a long day" = likely tension, not emergency
- "Sudden severe headache, worst of my life" = possible emergency

Return ONLY valid JSON (no markdown):
{{
    "urgency": "CRITICAL/HIGH/MODERATE/LOW",
    "reasoning": "Brief explanation of classification",
    "recommended_action": "What the patient should do",
    "initial_response": "A brief, empathetic response to the patient",
    "emergency_message": "If CRITICAL, the emergency guidance message"
}}"""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw = response["message"]["content"]
            cleaned = re.sub(r'<[^>]+>', '', raw).strip()
            
            # Try to parse JSON
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    # Default to moderate if parsing fails
                    return {
                        "urgency": UrgencyLevel.MODERATE.value,
                        "reasoning": "Unable to assess - defaulting to moderate",
                        "recommended_action": "Schedule an appointment with your doctor"
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error assessing urgency: {e}")
            return {
                "urgency": UrgencyLevel.MODERATE.value,
                "reasoning": "Error in assessment - defaulting to moderate",
                "recommended_action": "Schedule an appointment with your doctor"
            }
    
    def _extract_phi_from_text(self, text: str) -> None:
        """Extract PHI (age, sex) from text."""
        text_lower = text.lower()
        
        # Extract age
        age_patterns = [
            r'i am (\d{1,3})(?:\s*years?\s*old)?',
            r'(?:my\s+)?age\s+(?:is\s+)?(\d{1,3})',
            r'(\d{1,3})\s*years?\s*old',
            r'^(\d{1,3})$'
        ]
        for pattern in age_patterns:
            match = re.search(pattern, text_lower)
            if match:
                age = int(match.group(1))
                if 0 < age < 150:
                    self.clinical_data["patient_profile"]["age"] = age
                    break
        
        # Extract sex
        if re.search(r'\b(male|man|boy)\b', text_lower):
            self.clinical_data["patient_profile"]["sex"] = "male"
        elif re.search(r'\b(female|woman|girl)\b', text_lower):
            self.clinical_data["patient_profile"]["sex"] = "female"
    
    def _extract_symptoms_from_text(self, text: str) -> None:
        """Extract symptoms from text."""
        # Common symptom keywords
        symptom_keywords = [
            "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
            "pain", "dizziness", "rash", "swelling", "shortness of breath",
            "chest pain", "stomach pain", "back pain", "joint pain",
            "throat pain", "sore throat", "migraine", "body ache",
            "weakness", "chills", "sweating", "congestion", "runny nose"
        ]
        
        text_lower = text.lower()
        for symptom in symptom_keywords:
            if symptom in text_lower:
                if symptom not in [s.get("name") for s in self.clinical_data["symptoms"]]:
                    self.clinical_data["symptoms"].append({
                        "name": symptom,
                        "mentioned": True
                    })
    
    def _extract_clinical_info(self, text: str) -> None:
        """Extract clinical information from text using LLM."""
        
        prompt = f"""Extract clinical information from this patient response.

PATIENT RESPONSE: "{text}"

CURRENT CLINICAL DATA:
{json.dumps(self.clinical_data, indent=2)}

Extract and return JSON with any new information found:
{{
    "symptoms": ["any new symptoms mentioned"],
    "symptom_details": {{
        "symptom_name": {{
            "duration": "how long",
            "severity": "1-10 or description",
            "location": "where on body",
            "quality": "sharp, dull, burning, etc"
        }}
    }},
    "medical_history": ["any conditions mentioned"],
    "medications": ["any medications mentioned"],
    "allergies": ["any allergies mentioned"],
    "additional_notes": "any other relevant clinical information"
}}

Return ONLY valid JSON. If nothing found, return empty values."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw = response["message"]["content"]
            cleaned = re.sub(r'<[^>]+>', '', raw).strip()
            
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    return
            
            # Update clinical data
            for symptom in result.get("symptoms", []):
                if symptom and symptom not in [s.get("name") for s in self.clinical_data["symptoms"]]:
                    self.clinical_data["symptoms"].append({"name": symptom})
            
            # Update symptom details
            for symptom, details in result.get("symptom_details", {}).items():
                if symptom and details:
                    self.clinical_data["symptom_details"][symptom] = details
            
            # Update other fields
            for field in ["medical_history", "medications", "allergies"]:
                for item in result.get(field, []):
                    if item and item not in self.clinical_data[field]:
                        self.clinical_data[field].append(item)
            
        except Exception as e:
            logger.error(f"Error extracting clinical info: {e}")
    
    def _get_missing_phi(self) -> List[str]:
        """Get list of missing required PHI."""
        missing = []
        profile = self.clinical_data["patient_profile"]
        
        if profile["age"] is None:
            missing.append("age")
        if profile["sex"] is None:
            missing.append("sex")
        
        return missing
    
    def _generate_phi_question(self, missing: List[str]) -> str:
        """Generate a question for missing PHI."""
        if "age" in missing and "sex" not in missing:
            return "Could you tell me your age?"
        elif "sex" in missing and "age" not in missing:
            return "Could you tell me your biological sex (male or female)?"
        elif "age" in missing and "sex" in missing:
            return "To help me better assist you, could you tell me your age and biological sex?"
        return "Could you provide a bit more information about yourself?"
    
    def _should_request_image(self) -> bool:
        """Determine if image would aid triage."""
        # Don't request image if one has already been processed
        if self.image_processed:
            return False
        
        chief_complaint = self.clinical_data["chief_complaint"].lower()
        symptoms_text = " ".join([s.get("name", "") for s in self.clinical_data["symptoms"]]).lower()
        
        return any(indicator in chief_complaint or indicator in symptoms_text 
                   for indicator in self.VISUAL_INDICATORS)
    
    def _has_sufficient_info(self) -> bool:
        """Check if we have sufficient information for summary."""
        # Must have chief complaint
        if not self.clinical_data["chief_complaint"]:
            return False
        
        # Check for symptoms or symptom details
        has_symptoms = bool(self.clinical_data["symptoms"])
        has_symptom_details = bool(self.clinical_data["symptom_details"])
        
        # More lenient: proceed if we have symptoms OR symptom details
        # This handles cases where LLM extraction may have failed
        if not (has_symptoms or has_symptom_details):
            # Still allow proceeding if we have enough questions asked
            # The chief complaint itself contains symptom info
            if self.questions_asked < 3:
                return False
        
        # Reduce minimum questions to 2 for faster completion
        # This allows the conversation to progress more naturally
        return self.questions_asked >= 2
    
    def _generate_contextual_question(self) -> str:
        """Generate next contextual question using LLM."""
        
        # Build conversation summary to avoid repeating questions
        conversation_summary = self._build_conversation_summary()
        
        prompt = f"""You are a medical triage assistant. Generate ONE follow-up question for this patient.

CHIEF COMPLAINT: {self.clinical_data['chief_complaint']}

CURRENT SYMPTOMS: {json.dumps(self.clinical_data['symptoms'], indent=2)}

SYMPTOM DETAILS: {json.dumps(self.clinical_data['symptom_details'], indent=2)}

PATIENT PROFILE: {json.dumps(self.clinical_data['patient_profile'], indent=2)}

QUESTIONS ASKED SO FAR: {self.questions_asked}

RECENT CONVERSATION:
{conversation_summary}

IMPORTANT INSTRUCTIONS:
1. DO NOT repeat questions that have already been asked
2. Look at the conversation above to see what topics have been covered
3. Ask about NEW aspects that haven't been explored yet
4. If the patient has already answered questions about certain symptoms, move on to other topics
5. Consider asking about: medical history, medications, allergies, lifestyle factors, family history
6. If sufficient information has been gathered, you can ask if there's anything else they'd like to share

Generate a relevant follow-up question that explores NEW information.

Return ONLY the question text, nothing else. Make it conversational and empathetic."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            question = response["message"]["content"].strip()
            # Clean up any quotes or extra formatting
            question = question.strip('"\'').strip()
            
            return question
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return "Can you tell me more about how this has been affecting you?"
    
    def _generate_image_followup(self, image_analysis: str) -> str:
        """Generate follow-up questions based on image analysis."""
        
        prompt = f"""Based on this image analysis, generate follow-up questions for the patient.

IMAGE ANALYSIS: {image_analysis}

CHIEF COMPLAINT: {self.clinical_data['chief_complaint']}

Generate 1-2 relevant follow-up questions about what was observed in the image.
Keep questions conversational and empathetic.

Return ONLY the questions, nothing else."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Error generating image followup: {e}")
            return "Can you tell me more about when this started and how it's been changing?"
    
    def _generate_clinical_summary(self) -> str:
        """Generate pre-consultation clinical summary."""
        
        prompt = f"""Generate a pre-consultation summary for this patient.

CLINICAL DATA:
{json.dumps(self.clinical_data, indent=2)}

Create a structured summary including:
1. **Chief Complaint**: The main reason for the visit
2. **History of Present Illness**: Timeline, severity, associated symptoms
3. **Patient Profile**: Age, sex
4. **Relevant History**: Medical history, medications, allergies
5. **Assessment**: Summary of findings (NOT a diagnosis)
6. **Recommended Next Steps**: Based on urgency level

IMPORTANT:
- Do NOT provide a diagnosis
- Include appropriate medical disclaimers
- Be clear that this is for informational purposes only
- Suggest appropriate level of care based on urgency

Format the summary in a clear, readable way."""

        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = response["message"]["content"].strip()
            
            # Add disclaimer
            disclaimer = """

---
⚠️ **DISCLAIMER**: This summary is generated by an AI assistant and is NOT a medical diagnosis. 
Please consult with a qualified healthcare provider for proper medical advice, diagnosis, and treatment.
If you are experiencing a medical emergency, please call emergency services immediately."""
            
            return summary + disclaimer
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._generate_fallback_summary()
    
    def _generate_fallback_summary(self) -> str:
        """Generate a basic summary if LLM fails."""
        profile = self.clinical_data["patient_profile"]
        symptoms = ", ".join([s.get("name", "") for s in self.clinical_data["symptoms"]])
        
        return f"""
**Pre-Consultation Summary**

**Chief Complaint**: {self.clinical_data['chief_complaint']}

**Patient Profile**: {profile.get('age', 'Unknown')} years old, {profile.get('sex', 'Unknown')}

**Symptoms**: {symptoms or 'Not specified'}

**Urgency Level**: {self.clinical_data.get('urgency_level', 'Unknown')}

**Recommended Next Steps**: Please consult with a healthcare provider.

---
⚠️ **DISCLAIMER**: This summary is generated by an AI assistant and is NOT a medical diagnosis.
"""
    
    def _build_conversation_summary(self) -> str:
        """Build a summary of the conversation to avoid repeating questions."""
        if not self.message_history:
            return "No previous conversation."
        
        # Extract assistant messages (questions asked)
        assistant_messages = []
        user_messages = []
        
        for msg in self.message_history[-10:]:  # Last 10 messages
            if msg["role"] == "assistant":
                content = msg["content"]
                # Skip system messages
                if not content.startswith("You are a medical"):
                    assistant_messages.append(content)
            elif msg["role"] == "user":
                user_messages.append(msg["content"])
        
        summary = []
        summary.append("Patient responses and assistant questions:")
        for i, (user, assistant) in enumerate(zip(user_messages, assistant_messages)):
            summary.append(f"- Patient: {user[:100]}...")
            summary.append(f"  Assistant asked: {assistant[:150]}...")
        
        return "\n".join(summary) if summary else "No previous conversation."
    
    def _generate_response(
        self,
        message: str,
        state: TriageState = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response dict."""
        
        # Ensure message is never empty - provide fallback
        if not message or not message.strip():
            message = "I understand. Let me continue gathering some information to help you better."
        
        # Add assistant message to history
        self._add_message("assistant", message)
        
        response = {
            "message": message,
            "state": state.value if state else self.state.value,
            "session_id": self.session_id,
            "clinical_data": self.clinical_data,
            "questions_asked": self.questions_asked,
            **kwargs
        }
        
        return response
    
    def get_state(self) -> str:
        """Get current state."""
        return self.state.value
    
    def get_clinical_data(self) -> Dict[str, Any]:
        """Get current clinical data."""
        return self.clinical_data
    
    def get_message_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.message_history


# Factory function
def create_dynamic_triage_agent(
    session_id: str,
    llm_model: str = "gemini-3-flash-preview",
    rag_service=None,
    vision_processor=None
) -> DynamicTriageAgent:
    """Create a new dynamic triage agent."""
    return DynamicTriageAgent(
        session_id=session_id,
        llm_model=llm_model,
        rag_service=rag_service,
        vision_processor=vision_processor
    )
