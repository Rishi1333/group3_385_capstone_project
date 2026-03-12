"""
Tool Registry Service

Manages ML models as callable tools with defined schemas.
Each tool has required_data_fields and questions_to_ask for intake integration.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""
    name: str
    description: str
    parameters: Dict[str, str]  # parameter_name: description
    required_data_fields: List[str]  # Fields needed from intake
    questions_to_ask: List[Dict[str, str]]  # Questions to gather data
    callable: Optional[Callable] = None


class ToolRegistry:
    """
    Registry for managing ML models as callable tools.
    
    Each tool is defined with:
    - Name and description
    - Parameters schema
    - Required data fields for intake
    - Questions to ask during gathering phase
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._initialized = False
    
    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: ToolDefinition to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def register_callable(self, name: str, callable_func: Callable) -> None:
        """
        Register a callable function for a tool.
        
        Args:
            name: Tool name
            callable_func: Function to call
        """
        if name in self._tools:
            self._tools[name].callable = callable_func
        else:
            logger.warning(f"Tool {name} not found when registering callable")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            ToolDefinition or None
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> Dict[str, ToolDefinition]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary of tools
        """
        return self._tools.copy()
    
    def get_tools_for_symptoms(self, symptoms: List[str]) -> List[str]:
        """
        Determine which tools are relevant based on symptoms.
        
        Args:
            symptoms: List of symptoms
            
        Returns:
            List of tool names that could be relevant
        """
        # Map symptom categories to tools
        symptom_to_tool_map = {
            "diabetes": ["diabetes_risk"],
            "heart": ["heart_analysis"],
            "mental": ["mental_health_screen"],
            "general": ["symptom_predict"]
        }
        
        relevant_tools = set()
        symptoms_text = " ".join(symptoms).lower()
        
        for category, tools in symptom_to_tool_map.items():
            if category in symptoms_text:
                relevant_tools.update(tools)
        
        # Always include general symptom prediction
        relevant_tools.add("symptom_predict")
        
        return list(relevant_tools)
    
    def get_required_fields_for_tools(self, tool_names: List[str]) -> List[str]:
        """
        Get all required data fields for a set of tools.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            List of unique required field names
        """
        fields = set()
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                fields.update(tool.required_data_fields)
        return list(fields)
    
    def get_questions_for_tools(self, tool_names: List[str]) -> List[Dict[str, str]]:
        """
        Get all questions to ask for a set of tools.
        
        Args:
            tool_names: List of tool names
            
        Returns:
            List of question dicts
        """
        questions = []
        seen = set()
        
        for name in tool_names:
            tool = self.get_tool(name)
            if tool:
                for q in tool.questions_to_ask:
                    # Avoid duplicates
                    q_key = q.get("field", q.get("question", ""))
                    if q_key not in seen:
                        questions.append(q)
                        seen.add(q_key)
        
        return questions
    
    def invoke_tool(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool with the given data.
        
        Args:
            name: Tool name
            data: Data to pass to the tool
            
        Returns:
            Tool execution result
        """
        tool = self.get_tool(name)
        if not tool:
            return {"error": f"Tool {name} not found"}
        
        if not tool.callable:
            return {"error": f"Tool {name} has no callable registered"}
        
        try:
            result = tool.callable(data)
            return result
        except Exception as e:
            logger.error(f"Error invoking tool {name}: {e}")
            return {"error": str(e)}


# Predefined tool definitions
def get_default_tools() -> Dict[str, ToolDefinition]:
    """
    Get default tool definitions.
    
    Returns:
        Dictionary of default ToolDefinitions
    """
    return {
        "diabetes_risk": ToolDefinition(
            name="calculate_diabetes_risk",
            description="Calculate diabetes risk based on clinical measurements",
            parameters={
                "glucose": "Fasting glucose level (mg/dL)",
                "bmi": "Body Mass Index",
                "age": "Patient age",
                "blood_pressure": "Systolic blood pressure (mmHg)",
                "pregnancies": "Number of pregnancies (if applicable)",
                "insulin": "Insulin level",
                "cholesterol": "Total cholesterol"
            },
            required_data_fields=[
                "glucose", "bmi", "age", "blood_pressure"
            ],
            questions_to_ask=[
                {"field": "age", "question": "What is your age?"},
                {"field": "glucose", "question": "Do you know your fasting blood glucose level? If so, what is it (mg/dL)?"},
                {"field": "bmi", "question": "What is your weight and height? (We can calculate BMI)"},
                {"field": "blood_pressure", "question": "What is your blood pressure reading?"},
                {"field": "family_history", "question": "Do you have a family history of diabetes?"}
            ]
        ),
        
        "heart_analysis": ToolDefinition(
            name="analyze_heart_symptoms",
            description="Analyze cardiac risk based on symptoms and clinical features",
            parameters={
                "age": "Patient age",
                "sex": "Biological sex (male/female)",
                "chest_pain_type": "Type of chest pain",
                "resting_bp": "Resting blood pressure (mmHg)",
                "cholesterol": "Serum cholesterol (mg/dL)",
                "fasting_sugar": "Fasting blood sugar > 120 mg/dL",
                "ecg": "Resting ECG results",
                "max_heart_rate": "Maximum heart rate achieved",
                "exercise_angina": "Exercise-induced angina",
                "st_depression": "ST depression"
            },
            required_data_fields=[
                "age", "sex", "chest_pain_type", "resting_bp", "cholesterol"
            ],
            questions_to_ask=[
                {"field": "age", "question": "What is your age?"},
                {"field": "sex", "question": "What is your biological sex?"},
                {"field": "chest_pain_type", "question": "What type of chest pain do you experience? (typical angina, atypical angina, non-anginal, or none)"},
                {"field": "resting_bp", "question": "What is your resting blood pressure?"},
                {"field": "cholesterol", "question": "Do you know your cholesterol level?"},
                {"field": "max_heart_rate", "question": "What is your maximum heart rate during exercise (if known)?"}
            ]
        ),
        
        "symptom_predict": ToolDefinition(
            name="predict_disease_from_symptoms",
            description="Predict potential disease from symptom constellation",
            parameters={
                "symptoms": "Dictionary of symptom names to binary presence"
            },
            required_data_fields=["symptoms"],
            questions_to_ask=[
                {"field": "symptoms", "question": "Please describe all symptoms you are experiencing"}
            ]
        ),
        
        "mental_health_screen": ToolDefinition(
            name="screen_mental_health",
            description="Screen for mental health treatment needs",
            parameters={
                "work_interfere": "How much work is affected",
                "family_history": "Family history of mental health",
                "benefits": "Employer provides mental health benefits",
                "care_options": "Knows available care options",
                "treatment_seeking": "Seeking treatment"
            },
            required_data_fields=[
                "work_interfere", "family_history"
            ],
            questions_to_ask=[
                {"field": "work_interfere", "question": "How much does your mental health affect your work? (never, sometimes, often)"},
                {"field": "family_history", "question": "Do you have a family history of mental health conditions?"},
                {"field": "benefits", "question": "Does your employer provide mental health benefits?"},
                {"field": "care_options", "question": "Do you know what mental health care options are available to you?"}
            ]
        )
    }


# Singleton instance
_tool_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get or create the singleton ToolRegistry instance."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        # Register default tools
        for tool in get_default_tools().values():
            _tool_registry.register_tool(tool)
    return _tool_registry
