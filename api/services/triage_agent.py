"""
Triage Agent Service

Implements the state machine for patient intake flow.
Manages GATHERING -> ANALYZING -> REPORTING -> COMPLETE workflow.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import ollama

# Configure logging
logger = logging.getLogger(__name__)


# State definitions
class TriageState:
    IDLE = "IDLE"
    GATHERING = "GATHERING"
    ANALYZING = "ANALYZING"
    REPORTING = "REPORTING"
    COMPLETE = "COMPLETE"


class TriageAgent:
    """
    State machine agent for medical triage.
    
    Manages the patient intake workflow:
    - IDLE: Waiting for initial input
    - GATHERING: Asking follow-up questions
    - ANALYZING: Invoking tools for prediction
    - REPORTING: Generating clinical summary
    - COMPLETE: Session finished
    """
    
    def __init__(
        self,
        session_id: str,
        tool_registry=None,
        llm_model: str = "gemini-3-flash-preview"
    ):
        """
        Initialize the triage agent.
        
        Args:
            session_id: Unique session identifier
            tool_registry: ToolRegistry instance
            llm_model: LLM model for generating questions
        """
        self.session_id = session_id
        self.state = TriageState.IDLE
        self.llm_model = llm_model
        self.tool_registry = tool_registry
        
        # Clinical data storage
        self.clinical_data = {
            "chief_complaint": "",
            "symptoms": [],
            "medical_history": [],
            "medications": [],
            "risk_factors": [],
            "allergies": [],  # NEW: Track allergies
            "lifestyle": {  # NEW: Lifestyle factors
                "smoking": None,  # 'never', 'former', 'current'
                "alcohol": None,  # 'none', 'occasional', 'moderate', 'heavy'
                "exercise": None  # 'none', 'light', 'moderate', 'active'
            },
            "family_history_details": [],  # NEW: Detailed family history
            "recent_travel": None,  # NEW: Recent travel history
            "occupation": None,  # NEW: Patient occupation
            "image_findings": None,
            "patient_profile": {
                "age": None,
                "sex": None
            }
        }
        
        # Message history
        self.message_history = []
        
        # Intake tracking
        self.questions_asked = 0
        self.max_questions = 5
        self.required_fields_collected = set()
        self.tools_to_invoke = []
        
        # Track what information has been explicitly addressed (even if "none")
        self.info_addressed = {
            "medical_history": False,
            "medications": False,
            "risk_factors": False,
            "allergies": False,  # NEW
            "lifestyle_smoking": False,  # NEW
            "lifestyle_alcohol": False,  # NEW
            "lifestyle_exercise": False,  # NEW
            "family_history_details": False,  # NEW
            "recent_travel": False,  # NEW
            "occupation": False  # NEW
        }
        
        # Track which fields we've already asked about (prevents repeating questions)
        self.asked_fields = set()
        
        # Add system message
        self._add_message("system", self._get_system_prompt())
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the triage agent."""
        return """You are a medical triage assistant. Your role is to:
1. Gather patient symptoms and medical history through conversational questions
2. Be empathetic, clear, and professional
3. Ask one question at a time to avoid overwhelming the patient
4. Collect necessary clinical information for assessment
5. NEVER provide diagnosis - only gather information

Guidelines:
- Ask about symptom duration, severity, and progression
- Inquire about relevant medical history
- Ask about current medications
- Note any risk factors (family history, lifestyle)
- After gathering sufficient info, transition to analysis phase

Do not mention that you are an AI or use technical jargon with patients."""
    
    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the history."""
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def start_intake(self, initial_text: str) -> Dict[str, Any]:
        """
        Start the intake process with initial patient input.
        
        Args:
            initial_text: Patient's initial description
            
        Returns:
            Response dict with next action
        """
        self.state = TriageState.GATHERING
        self.clinical_data["chief_complaint"] = initial_text
        
        # Extract any immediate symptoms mentioned
        self._extract_symptoms_from_text(initial_text)
        
        # Extract age, sex, and other clinical data from initial message
        self._extract_data_from_response(initial_text)
        
        # Add user message to history
        self._add_message("user", initial_text)
        
        # Generate first question
        return self._generate_next_question()
    
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process a user message during gathering phase.
        
        Args:
            user_message: User's response to a question
            
        Returns:
            Response dict with next action
        """
        if self.state != TriageState.GATHERING:
            return {
                "error": f"Cannot process message in {self.state} state"
            }
        
        # Add to history
        self._add_message("user", user_message)
        
        # Extract clinical data from response
        self._extract_data_from_response(user_message)
        
        # Check if we have enough info to analyze
        if self._should_transition_to_analysis():
            return self._transition_to_analysis()
        
        # Check if max questions reached
        if self.questions_asked >= self.max_questions:
            return self._transition_to_analysis()
        
        # Generate next question
        return self._generate_next_question()
    
    def _extract_symptoms_from_text(self, text: str) -> None:
        """Extract mentioned symptoms from text."""
        # Common symptom keywords
        symptom_keywords = [
            "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
            "pain", "dizziness", "rash", "swelling", "shortness of breath",
            "chest pain", "stomach pain", "back pain", "joint pain",
            "throat pain", "sore throat", "headache", "migraine",
            "body ache", "muscle pain", "weakness", "chills", "sweating",
            "congestion", "runny nose", "sneezing", "itchy eyes", "watery eyes"
        ]
        
        text_lower = text.lower()
        for symptom in symptom_keywords:
            if symptom in text_lower:
                # Avoid duplicates
                if not any(s.get("name") == symptom for s in self.clinical_data["symptoms"]):
                    self.clinical_data["symptoms"].append({
                        "name": symptom,
                        "mentioned": True,
                        "duration": None,
                        "severity": None
                    })
    
    def _extract_data_from_response(self, response: str) -> None:
        """Extract structured data from user response."""
        # First try simple regex-based extraction (fallback)
        self._extract_data_with_regex(response)
        
        # Then try LLM for more complex extraction
        prompt = f"""
Extract clinical data from the patient response.

RESPONSE: "{response}"

EXISTING DATA:
{json.dumps(self.clinical_data, indent=2)}

Extract and return JSON with:
- age: number if mentioned
- sex: "male" or "female" if mentioned
- symptoms: add any new symptoms found (append to existing)
- medical_history: any conditions mentioned
- medications: any medications mentioned
- risk_factors: any risk factors mentioned
- duration: symptom duration if mentioned
- severity: severity level if mentioned (1-10)

Return ONLY valid JSON.
"""
        
        try:
            response_msg = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw = response_msg["message"]["content"]
            cleaned = re.sub(r'<[^>]+>', '', raw).strip()
            
            try:
                extracted = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                if match:
                    extracted = json.loads(match.group(0))
                else:
                    return
            
            # Update clinical data
            if extracted.get("age"):
                self.clinical_data["patient_profile"]["age"] = extracted["age"]
            if extracted.get("sex"):
                self.clinical_data["patient_profile"]["sex"] = extracted["sex"]
            
            # Add new symptoms
            for symptom in extracted.get("symptoms", []):
                # Handle both dict and string format
                if isinstance(symptom, dict):
                    symptom_name = symptom.get("name", "")
                else:
                    symptom_name = str(symptom)
                    symptom = {"name": symptom_name}
                
                if symptom_name and not any(s.get("name") == symptom_name for s in self.clinical_data["symptoms"]):
                    self.clinical_data["symptoms"].append(symptom)
            
            # Add medical history
            for history in extracted.get("medical_history", []):
                if history not in self.clinical_data["medical_history"]:
                    self.clinical_data["medical_history"].append(history)
            
            # Add medications
            for med in extracted.get("medications", []):
                if med not in self.clinical_data["medications"]:
                    self.clinical_data["medications"].append(med)
            
            # Add risk factors
            for risk in extracted.get("risk_factors", []):
                if risk not in self.clinical_data["risk_factors"]:
                    self.clinical_data["risk_factors"].append(risk)
                    
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
    
    def _extract_data_with_regex(self, response: str) -> None:
        """Extract data using simple regex patterns (fallback when LLM unavailable)."""
        response_lower = response.lower()
        
        # Check for comprehensive "no" answers that cover multiple categories
        # If user says something like "no medication or supplements, no family history or lifestyle"
        # they're indicating they have no relevant medical background at all
        comprehensive_no_patterns = [
            r'\bno\s+(medication|meds?|supplements?).*no\s+(family|lifestyle|history)',
            r'\bno\s+(family|lifestyle|history).*no\s+(medication|meds?|supplements?)',
            r'\bnothing\s+(to\s+report|significant|relevant|notable)',
            r'\bi\s+(don\'?t|do\s+not)\s+have\s+any',
            r'\bno\s+medical\s+(history|conditions?|issues?|problems?|concerns?)',
        ]
        
        for pattern in comprehensive_no_patterns:
            if re.search(pattern, response_lower):
                # Mark all categories as addressed since user gave comprehensive "no"
                self.info_addressed["medical_history"] = True
                self.info_addressed["medications"] = True
                self.info_addressed["risk_factors"] = True
                logger.info(f"Comprehensive 'no' detected, marking all info as addressed")
                return  # Early return since all categories are addressed
        
        # Check for "no X" patterns to mark fields as addressed
        # Medical history - expanded patterns
        if (re.search(r'\bno\s+(medical\s+)?history\b', response_lower) or
            re.search(r'\bno\s+(medical\s+)?conditions?\b', response_lower) or
            re.search(r'\bno\s+(past|existing|current)\s+(medical\s+)?(conditions?|history|issues?)\b', response_lower) or
            re.search(r'\b(none|no)\s+(of\s+)?(the\s+)?above\b', response_lower) or
            re.search(r'\bno\s+health\s+(issues?|problems?|concerns?)\b', response_lower)):
            self.info_addressed["medical_history"] = True
        
        # Medications - expanded patterns
        if (re.search(r'\bno\s+medications?\b', response_lower) or
            re.search(r'\bno\s+(meds|medicine)\b', response_lower) or
            re.search(r'\bnot\s+(on|taking)\s+any\s+(medication|meds|medicine)', response_lower) or
            re.search(r'\bno\s+prescription', response_lower)):
            self.info_addressed["medications"] = True
        
        # Supplements - also marks medications as addressed
        if re.search(r'\bno\s+supplements?\b', response_lower):
            self.info_addressed["medications"] = True
        
        # Risk factors / family history / lifestyle - expanded patterns
        if (re.search(r'\bno\s+(family\s+)?history\b', response_lower) or
            re.search(r'\bno\s+risk\s+factors?\b', response_lower) or
            re.search(r'\bno\s+family\s+(medical\s+)?(history|background)\b', response_lower) or
            re.search(r'\bno\s+lifestyle\b', response_lower) or
            re.search(r'\bno\s+(known\s+)?family\s+(health\s+)?(issues?|problems?|concerns?)\b', response_lower)):
            self.info_addressed["risk_factors"] = True
        
        # NEW: Allergies patterns
        if (re.search(r'\bno\s+allergies?\b', response_lower) or
            re.search(r'\bnot\s+allergic\s+to\s+anything\b', response_lower) or
            re.search(r'\bno\s+known\s+allergies?\b', response_lower)):
            self.info_addressed["allergies"] = True
        
        # NEW: Smoking patterns
        if re.search(r'\b(i\s+don\'?t|do\s+not|never)\s+(smoke|smoked|use\s+tobacco)\b', response_lower):
            self.clinical_data["lifestyle"]["smoking"] = "never"
            self.info_addressed["lifestyle_smoking"] = True
        elif re.search(r'\b(former|ex-?smoker|quit\s+smoking)\b', response_lower):
            self.clinical_data["lifestyle"]["smoking"] = "former"
            self.info_addressed["lifestyle_smoking"] = True
        elif re.search(r'\b(currently?\s+)?smoke|smoking|tobacco\b', response_lower):
            self.clinical_data["lifestyle"]["smoking"] = "current"
            self.info_addressed["lifestyle_smoking"] = True
        
        # NEW: Exercise patterns
        if re.search(r'\b(no|don\'?t|do\s+not)\s+(exercise|work\s+out|workout)\b', response_lower):
            self.clinical_data["lifestyle"]["exercise"] = "none"
            self.info_addressed["lifestyle_exercise"] = True
        elif re.search(r'\b(light|occasional)\s*(exercise|activity)?\b', response_lower):
            self.clinical_data["lifestyle"]["exercise"] = "light"
            self.info_addressed["lifestyle_exercise"] = True
        elif re.search(r'\b(moderate|regular)\s*(exercise|activity)?\b', response_lower):
            self.clinical_data["lifestyle"]["exercise"] = "moderate"
            self.info_addressed["lifestyle_exercise"] = True
        elif re.search(r'\b(active|daily|every\s+day|5\s*[+-]\s*times?)\b', response_lower):
            self.clinical_data["lifestyle"]["exercise"] = "active"
            self.info_addressed["lifestyle_exercise"] = True
        
        # NEW: Occupation patterns - extract occupation
        occupation_match = re.search(r'\bi\s+(am|work)\s+(?:a\s+)?(\w+(?:\s+\w+)?)(?:\s+(?:by\s+profession|by\s+trade))?', response_lower)
        if occupation_match:
            self.clinical_data["occupation"] = occupation_match.group(2).title()
            self.info_addressed["occupation"] = True
        
        # NEW: Travel patterns
        if re.search(r'\bno\s+(recent\s+)?travel\b', response_lower) or re.search(r'\bhaven\'?t\s+traveled\b', response_lower):
            self.clinical_data["recent_travel"] = "none"
            self.info_addressed["recent_travel"] = True
        elif re.search(r'\btraveled\s+to\s+(\w+(?:\s+\w+)?)\b', response_lower):
            travel_match = re.search(r'\btraveled\s+to\s+(\w+(?:\s+\w+)?)\b', response_lower)
            self.clinical_data["recent_travel"] = travel_match.group(1).title()
            self.info_addressed["recent_travel"] = True
        
        # Extract age (look for patterns like "I am 35", "35 years old", "age is 35")
        age_patterns = [
            r'i am (\d{1,3})(?:\s*years?\s*old)?',
            r'(?:my\s+)?age\s+(?:is\s+)?(\d{1,3})',
            r'(\d{1,3})\s*years?\s*old',
            r'^(\d{1,3})$'  # Just a number
        ]
        for pattern in age_patterns:
            match = re.search(pattern, response_lower)
            if match:
                age = int(match.group(1))
                if 0 < age < 150:
                    self.clinical_data["patient_profile"]["age"] = age
                    break
        
        # Extract sex
        if re.search(r'\b(male|man|boy)\b', response_lower):
            self.clinical_data["patient_profile"]["sex"] = "male"
        elif re.search(r'\b(female|woman|girl)\b', response_lower):
            self.clinical_data["patient_profile"]["sex"] = "female"
        
        # Extract duration (look for patterns like "2 days", "for a week")
        duration_patterns = [
            r'(\d+)\s*(day|days|week|weeks|month|months|hour|hours)',
            r'(?:for\s+)?(?:a|an)\s+(day|week|month|hour)',
            r'(?:since\s+)(yesterday|last\s+week|this\s+morning)'
        ]
        for pattern in duration_patterns:
            match = re.search(pattern, response_lower)
            if match and self.clinical_data["symptoms"]:
                # Update the last symptom's duration
                last_symptom = self.clinical_data["symptoms"][-1]
                if isinstance(last_symptom, str):
                    # Convert string to dict
                    self.clinical_data["symptoms"][-1] = {
                        "name": last_symptom,
                        "duration": match.group(0),
                        "severity": None
                    }
                else:
                    last_symptom["duration"] = match.group(0)
                break
        
        # Extract severity (look for patterns like "7 out of 10", "severity 5")
        severity_match = re.search(r'(\d)(?:\s*(?:out\s+of|\/)\s*10)?', response_lower)
        if severity_match and self.clinical_data["symptoms"]:
            severity = int(severity_match.group(1))
            if 1 <= severity <= 10:
                last_symptom = self.clinical_data["symptoms"][-1]
                if isinstance(last_symptom, str):
                    # Convert string to dict
                    self.clinical_data["symptoms"][-1] = {
                        "name": last_symptom,
                        "duration": None,
                        "severity": severity
                    }
                else:
                    last_symptom["severity"] = severity
    
    def _should_transition_to_analysis(self) -> bool:
        """Determine if we have enough data to transition to analysis."""
        # Check if we have chief complaint and at least one symptom
        if not self.clinical_data["chief_complaint"]:
            logger.info("Not transitioning: no chief complaint")
            return False
        
        if not self.clinical_data["symptoms"]:
            logger.info("Not transitioning: no symptoms")
            return False
        
        # If we've asked about all basic fields, transition regardless of answers
        # This prevents infinite loops when users don't answer questions
        # Updated to include new fields
        basic_fields = {"age", "sex", "medical_history", "medications", "allergies", "risk_factors"}
        if self.asked_fields >= basic_fields:
            logger.info("Transitioning to analysis: all basic fields have been asked")
            return True
        
        # Check if we have key profile info
        profile = self.clinical_data["patient_profile"]
        if profile["age"] is None or profile["sex"] is None:
            logger.info(f"Not transitioning: missing profile info - age: {profile['age']}, sex: {profile['sex']}")
            return False
        
        # Check if all required info categories have been addressed
        # (either has data or user confirmed "none")
        # Updated to include new fields
        all_addressed = (
            (self.clinical_data["medical_history"] or self.info_addressed["medical_history"]) and
            (self.clinical_data["medications"] or self.info_addressed["medications"]) and
            (self.clinical_data["risk_factors"] or self.info_addressed["risk_factors"]) and
            (self.clinical_data.get("allergies") or self.info_addressed.get("allergies", False))
        )
        
        logger.info(f"Transition check: all_addressed={all_addressed}, info_addressed={self.info_addressed}")
        
        # If all categories addressed, we can proceed
        if all_addressed:
            logger.info("Transitioning to analysis: all categories addressed")
            return True
        
        # Check if we have at least 3 pieces of clinical info
        info_count = (
            len(self.clinical_data["symptoms"]) +
            len(self.clinical_data["medical_history"]) +
            len(self.clinical_data["medications"]) +
            len(self.clinical_data["risk_factors"]) +
            len(self.clinical_data.get("allergies", []))
        )
        
        logger.info(f"Transition check: info_count={info_count}")
        return info_count >= 3
    
    def _generate_next_question(self) -> Dict[str, Any]:
        """Generate the next question to ask the patient."""
        # Determine what we still need - skip fields we've already asked about
        missing = []
        
        # Basic profile info
        if self.clinical_data["patient_profile"]["age"] is None and "age" not in self.asked_fields:
            missing.append("age")
        if self.clinical_data["patient_profile"]["sex"] is None and "sex" not in self.asked_fields:
            missing.append("sex")
        
        # Medical history and medications
        if not self.clinical_data["medical_history"] and not self.info_addressed["medical_history"] and "medical_history" not in self.asked_fields:
            missing.append("medical_history")
        if not self.clinical_data["medications"] and not self.info_addressed["medications"] and "medications" not in self.asked_fields:
            missing.append("medications")
        
        # NEW: Allergies
        if not self.clinical_data["allergies"] and not self.info_addressed["allergies"] and "allergies" not in self.asked_fields:
            missing.append("allergies")
        
        # Risk factors and family history
        if not self.clinical_data["risk_factors"] and not self.info_addressed["risk_factors"] and "risk_factors" not in self.asked_fields:
            missing.append("risk_factors")
        if not self.clinical_data["family_history_details"] and not self.info_addressed["family_history_details"] and "family_history_details" not in self.asked_fields:
            missing.append("family_history_details")
        
        # NEW: Lifestyle factors
        lifestyle = self.clinical_data.get("lifestyle", {})
        if lifestyle.get("smoking") is None and not self.info_addressed.get("lifestyle_smoking") and "lifestyle_smoking" not in self.asked_fields:
            missing.append("lifestyle_smoking")
        if lifestyle.get("exercise") is None and not self.info_addressed.get("lifestyle_exercise") and "lifestyle_exercise" not in self.asked_fields:
            missing.append("lifestyle_exercise")
        
        # NEW: Occupation and travel
        if self.clinical_data.get("occupation") is None and not self.info_addressed.get("occupation") and "occupation" not in self.asked_fields:
            missing.append("occupation")
        if self.clinical_data.get("recent_travel") is None and not self.info_addressed.get("recent_travel") and "recent_travel" not in self.asked_fields:
            missing.append("recent_travel")
        
        # Debug logging
        logger.info(f"Generating next question. Missing: {missing}, Info addressed: {self.info_addressed}, "
                    f"Clinical data: {self.clinical_data}, Questions asked: {self.questions_asked}")
        
        # If we have profile info, ask about symptoms in more detail
        if not missing or self.questions_asked >= 2:
            # Find a symptom that hasn't been asked about yet
            if self.clinical_data["symptoms"]:
                for symptom in self.clinical_data["symptoms"]:
                    # Handle both dict and string format
                    if isinstance(symptom, dict):
                        symptom_name = symptom.get('name', 'this symptom')
                        # Check if this symptom already has duration info OR was already asked about
                        if symptom.get('duration') or f"duration:{symptom_name}" in self.asked_fields:
                            continue  # Skip symptoms that already have duration or were asked
                    else:
                        symptom_name = str(symptom)
                        if f"duration:{symptom_name}" in self.asked_fields:
                            continue
                    
                    # Mark this symptom duration as asked
                    self.asked_fields.add(f"duration:{symptom_name}")
                    
                    # Ask about this symptom
                    question = f"How long have you been experiencing {symptom_name}?"
                    self.questions_asked += 1
                    return {
                        "state": self.state,
                        "question": question,
                        "question_type": "duration",
                        "context": symptom
                    }
        
        # Generate question for missing info
        # Get symptom names safely
        symptom_names = []
        for s in self.clinical_data['symptoms']:
            if isinstance(s, dict):
                symptom_names.append(s.get('name', str(s)))
            else:
                symptom_names.append(str(s))
        
        question_prompt = f"""
Generate a patient-friendly question to gather the following missing information:
{missing}

CURRENT CONTEXT:
- Chief complaint: {self.clinical_data['chief_complaint']}
- Symptoms: {symptom_names}

Return ONLY the question as a single sentence.
"""
        
        # If no missing fields remain, transition to analysis
        if not missing:
            return self._transition_to_analysis()
        
        # Mark the first missing field as asked before generating the question
        field_to_ask = missing[0]
        self.asked_fields.add(field_to_ask)
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": question_prompt}]
            )
            
            question = response["message"]["content"].strip()
            question = re.sub(r'<[^>]+>', '', question)
            
            self.questions_asked += 1
            
            return {
                "state": self.state,
                "question": question,
                "missing_info": missing
            }
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            # Fallback questions including new fields
            fallback_questions = {
                "age": "What is your age?",
                "sex": "What is your biological sex?",
                "medical_history": "Do you have any existing medical conditions?",
                "medications": "Are you currently taking any medications?",
                "allergies": "Do you have any known allergies (medications, food, or environmental)?",
                "risk_factors": "Do you have any lifestyle risk factors (smoking, alcohol use)?",
                "family_history_details": "Does anyone in your immediate family have any chronic conditions like diabetes, heart disease, or cancer?",
                "lifestyle_smoking": "Do you currently smoke or use tobacco products?",
                "lifestyle_exercise": "How often do you exercise per week? (none, light, moderate, or active)",
                "occupation": "What is your occupation?",
                "recent_travel": "Have you traveled outside your usual area in the past 2 weeks?"
            }
            
            for key in missing:
                if key in fallback_questions:
                    self.questions_asked += 1
                    return {
                        "state": self.state,
                        "question": fallback_questions[key],
                        "missing_info": [key]
                    }
            
            return {
                "state": self.state,
                "question": "Can you tell me more about your symptoms?",
                "missing_info": []
            }
    
    def _transition_to_analysis(self) -> Dict[str, Any]:
        """Transition from gathering to analyzing state."""
        self.state = TriageState.ANALYZING
        self._add_message("system", "Transitioning to analysis phase. Sufficient information gathered.")
        
        # Determine which tools to invoke based on symptoms
        # Handle both dict format {"name": "symptom"} and string format "symptom"
        symptoms_text = []
        for s in self.clinical_data["symptoms"]:
            if isinstance(s, dict):
                symptoms_text.append(s.get("name", str(s)))
            else:
                symptoms_text.append(str(s))
        self.tools_to_invoke = self.tool_registry.get_tools_for_symptoms(symptoms_text) if self.tool_registry else []
        
        return {
            "state": self.state,
            "message": "Thank you for that information. I'm now analyzing your symptoms.",
            "tools_to_invoke": self.tools_to_invoke,
            "clinical_data": self.clinical_data
        }
    
    def set_image_findings(self, findings: str) -> None:
        """Set image analysis findings."""
        self.clinical_data["image_findings"] = findings
    
    def get_state(self) -> str:
        """Get current state."""
        return self.state
    
    def get_clinical_data(self) -> Dict:
        """Get collected clinical data."""
        return self.clinical_data
    
    def get_message_history(self) -> List[Dict]:
        """Get message history."""
        return self.message_history.copy()
    
    def to_dict(self) -> Dict:
        """Convert agent state to dictionary for session storage."""
        return {
            "session_id": self.session_id,
            "state": self.state,
            "clinical_data": self.clinical_data,
            "message_history": self.message_history,
            "questions_asked": self.questions_asked,
            "tools_to_invoke": self.tools_to_invoke,
            "info_addressed": self.info_addressed,
            "asked_fields": list(self.asked_fields)
        }
    
    @classmethod
    def from_dict(cls, data: Dict, tool_registry=None) -> "TriageAgent":
        """Restore agent from dictionary."""
        agent = cls(
            session_id=data.get("session_id", ""),
            tool_registry=tool_registry
        )
        agent.state = data.get("state", TriageState.IDLE)
        agent.clinical_data = data.get("clinical_data", agent.clinical_data)
        agent.message_history = data.get("message_history", [])
        agent.questions_asked = data.get("questions_asked", 0)
        agent.tools_to_invoke = data.get("tools_to_invoke", [])
        agent.info_addressed = data.get("info_addressed", agent.info_addressed)
        agent.asked_fields = set(data.get("asked_fields", []))
        return agent
