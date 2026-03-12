"""
Clinical Report Generator Service

Generates structured clinical summary reports from triage data.
"""

import json
import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import ollama

# Configure logging
logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    """
    Generates structured clinical summary reports for doctor review.
    """
    
    def __init__(self, llm_model: str = "gemini-3-flash-preview"):
        """
        Initialize the report generator.
        
        Args:
            llm_model: LLM model for generating reports
        """
        self.llm_model = llm_model
    
    def generate_report(
        self,
        clinical_data: Dict,
        tool_results: List[Dict] = None,
        image_findings: str = None
    ) -> Dict[str, Any]:
        """
        Generate a clinical summary report.
        
        Args:
            clinical_data: Collected clinical data from triage
            tool_results: Results from invoked tools
            image_findings: Image analysis findings (if any)
            
        Returns:
            Structured clinical report
        """
        tool_results = tool_results or []
        
        # Build prompt for report generation
        prompt = self._build_report_prompt(clinical_data, tool_results, image_findings)
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            report_text = response["message"]["content"]
            report_text = re.sub(r'<[^>]+>', '', report_text).strip()
            
            # Parse structured data from response
            return self._parse_report(report_text, clinical_data, tool_results)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return self._generate_fallback_report(clinical_data, tool_results)
    
    def _build_report_prompt(
        self,
        clinical_data: Dict,
        tool_results: List[Dict],
        image_findings: str
    ) -> str:
        """Build the prompt for report generation."""
        
        # Format clinical data
        patient_profile = clinical_data.get("patient_profile", {})
        symptoms = clinical_data.get("symptoms", [])
        medical_history = clinical_data.get("medical_history", [])
        medications = clinical_data.get("medications", [])
        risk_factors = clinical_data.get("risk_factors", [])
        chief_complaint = clinical_data.get("chief_complaint", "")
        
        # Format tool results
        tool_summary = ""
        for result in tool_results:
            tool_name = result.get("tool", "unknown")
            prediction = result.get("prediction", "N/A")
            tool_summary += f"\n- {tool_name}: {prediction}"
        
        prompt = f"""
Generate a clinical summary report for a doctor's review based on the following patient information:

PATIENT PROFILE:
- Age: {patient_profile.get('age', 'Not provided')}
- Sex: {patient_profile.get('sex', 'Not provided')}

CHIEF COMPLAINT:
{chief_complaint}

SYMPTOMS:
{json.dumps(symptoms, indent=2)}

MEDICAL HISTORY:
{json.dumps(medical_history, indent=2) if medical_history else "None reported"}

CURRENT MEDICATIONS:
{json.dumps(medications, indent=2) if medications else "None reported"}

RISK FACTORS:
{json.dumps(risk_factors, indent=2) if risk_factors else "None identified"}

TOOL ANALYSIS RESULTS:
{tool_summary if tool_summary else "No tool analysis available"}

{"IMAGE FINDINGS:" + image_findings if image_findings else ""}

Generate a structured clinical report with the following sections:

1. PATIENT PROFILE - Summary of demographics
2. CHIEF COMPLAINT - Main concern in medical terminology
3. HISTORY OF PRESENT ILLNESS - Timeline of symptoms
4. RISK FACTORS - Identified risk factors
5. VISUAL FINDINGS - Image analysis (if applicable)
6. SUGGESTED DIFFERENTIAL - Potential conditions (NOT definitive diagnoses)
7. RECOMMENDED NEXT STEPS - Lab tests, specialist referrals, or home care

IMPORTANT:
- Use proper medical terminology
- Suggested differentials should be likelihood-based (High/Medium/Low)
- Include disclaimer that this is AI assistance, NOT a medical diagnosis
- Keep recommendations practical and actionable

Return the report in a structured format that a doctor can quickly review.
"""
        return prompt
    
    def _parse_report(
        self,
        report_text: str,
        clinical_data: Dict,
        tool_results: List[Dict]
    ) -> Dict[str, Any]:
        """Parse and structure the generated report."""
        
        # Extract sections using simple parsing
        sections = {
            "patient_profile": {},
            "chief_complaint": "",
            "history_of_present_illness": "",
            "risk_factors": [],
            "visual_findings": None,
            "suggested_differential": [],
            "recommended_next_steps": []
        }
        
        # Extract patient profile
        profile = clinical_data.get("patient_profile", {})
        sections["patient_profile"] = {
            "age": profile.get("age"),
            "sex": profile.get("sex")
        }
        
        # Extract sections from text
        current_section = None
        section_content = []
        
        for line in report_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            lower_line = line.lower()
            if 'patient profile' in lower_line or 'demographic' in lower_line:
                current_section = "patient_profile"
                continue
            elif 'chief complaint' in lower_line:
                current_section = "chief_complaint"
                continue
            elif 'history of present illness' in lower_line or 'present illness' in lower_line:
                current_section = "history_of_present_illness"
                continue
            elif 'risk factor' in lower_line:
                current_section = "risk_factors"
                continue
            elif 'visual finding' in lower_line or 'image' in lower_line:
                current_section = "visual_findings"
                continue
            elif 'differential' in lower_line:
                current_section = "suggested_differential"
                continue
            elif 'next step' in lower_line or 'recommend' in lower_line:
                current_section = "recommended_next_steps"
                continue
            
            # Add content to current section
            if current_section:
                section_content.append(line)
        
        # Fill in sections
        sections["chief_complaint"] = clinical_data.get("chief_complaint", "")
        sections["risk_factors"] = clinical_data.get("risk_factors", [])
        sections["visual_findings"] = clinical_data.get("image_findings")
        
        # Parse differential and recommendations from tool results
        for result in tool_results:
            pred = result.get("prediction", {})
            if isinstance(pred, dict):
                if "predictions" in pred:
                    sections["suggested_differential"].append({
                        "condition": pred["predictions"],
                        "likelihood": "Medium",
                        "source": result.get("tool", "unknown")
                    })
        
        # Build final report
        return {
            "report_id": f"RPT-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.utcnow().isoformat(),
            "patient_profile": sections["patient_profile"],
            "chief_complaint": sections["chief_complaint"],
            "history_of_present_illness": self._format_history(clinical_data),
            "risk_factors": sections["risk_factors"],
            "visual_findings": sections["visual_findings"],
            "suggested_differential": sections["suggested_differential"],
            "recommended_next_steps": self._extract_recommendations(report_text),
            "disclaimer": self._get_disclaimer()
        }
    
    def _format_history(self, clinical_data: Dict) -> str:
        """Format the history of present illness."""
        symptoms = clinical_data.get("symptoms", [])
        if not symptoms:
            return "Patient presents with reported symptoms."
        
        history = "Patient reports: "
        symptom_list = []
        for s in symptoms:
            name = s.get("name", "unknown symptom")
            duration = s.get("duration", "unknown duration")
            symptom_list.append(f"{name} (duration: {duration})")
        
        history += ", ".join(symptom_list)
        return history
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from report text."""
        recommendations = []
        
        # Common recommendation patterns
        patterns = [
            r"(?:refer|consult|see|visit)\s+(?:a\s+)?(\w+\s+)?(\w+\s+)?(?:specialist|doctor|physician)",
            r"(?:consider|order|get)\s+(\w+\s+)?(?:lab|blood|test|imaging|scan)",
            r"(?:monitor|track|watch)\s+",
            r"(?:rest|hydrate|avoid|follow-up)"
        ]
        
        # Simple extraction based on keywords
        lines = text.split('\n')
        for line in lines:
            lower = line.lower()
            if any(p in lower for p in ['recommend', 'suggest', 'consider', 'should']):
                if len(line) > 20 and len(line) < 200:
                    recommendations.append(line.strip())
        
        # Fallback recommendations if none found
        if not recommendations:
            recommendations = [
                "Follow up with primary care physician",
                "Monitor symptoms and seek care if they worsen",
                "Consider lifestyle modifications as discussed"
            ]
        
        return recommendations[:5]  # Limit to 5
    
    def _get_disclaimer(self) -> str:
        """Get the standard disclaimer."""
        return """
DISCLAIMER: This clinical summary report is generated by an AI assistant 
and is intended for informational purposes only. It is NOT a medical diagnosis 
and should NOT replace professional medical advice from a licensed healthcare 
provider. The suggested differentials are based on limited information and may 
not represent the actual condition. Please consult with a qualified healthcare 
professional for proper evaluation, diagnosis, and treatment.

If you are experiencing a medical emergency, please call emergency services immediately.
""".strip()
    
    def _generate_fallback_report(
        self,
        clinical_data: Dict,
        tool_results: List[Dict]
    ) -> Dict[str, Any]:
        """Generate a fallback report if LLM fails."""
        
        profile = clinical_data.get("patient_profile", {})
        
        # Build tool predictions
        differentials = []
        for result in tool_results:
            pred = result.get("prediction", {})
            if isinstance(pred, dict) and "predictions" in pred:
                differentials.append({
                    "condition": pred["predictions"],
                    "likelihood": "Unknown",
                    "source": result.get("tool", "unknown")
                })
        
        return {
            "report_id": f"RPT-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "generated_at": datetime.utcnow().isoformat(),
            "patient_profile": {
                "age": profile.get("age"),
                "sex": profile.get("sex")
            },
            "chief_complaint": clinical_data.get("chief_complaint", ""),
            "history_of_present_illness": f"Patient presents with: {', '.join([s['name'] for s in clinical_data.get('symptoms', [])])}",
            "risk_factors": clinical_data.get("risk_factors", []),
            "visual_findings": clinical_data.get("image_findings"),
            "suggested_differential": differentials,
            "recommended_next_steps": [
                "Consult with healthcare provider",
                "Monitor symptoms",
                "Seek immediate care if symptoms worsen"
            ],
            "disclaimer": self._get_disclaimer()
        }
    
    def format_for_display(self, report: Dict) -> str:
        """
        Format the report for display to the user.
        
        Args:
            report: Structured report dictionary
            
        Returns:
            Formatted markdown string with professional layout
        """
        lines = []
        
        # Header
        lines.append("# CLINICAL SUMMARY REPORT")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Report metadata
        lines.append(f"**Report ID:** `{report.get('report_id', 'N/A')}`")
        lines.append("")
        lines.append(f"**Generated:** {report.get('generated_at', 'N/A')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Patient Profile - Table format
        profile = report.get("patient_profile", {})
        lines.append("## Patient Profile")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        lines.append(f"| Age | {profile.get('age', 'Not provided')} |")
        lines.append(f"| Sex | {profile.get('sex', 'Not provided').title() if profile.get('sex') else 'Not provided'} |")
        lines.append("")
        
        # Chief Complaint
        lines.append("## Chief Complaint")
        lines.append("")
        chief = report.get("chief_complaint", "Not provided")
        lines.append(f"> {chief}")
        lines.append("")
        
        # History of Present Illness
        lines.append("## History of Present Illness")
        lines.append("")
        history = report.get("history_of_present_illness", "Not provided")
        lines.append(history)
        lines.append("")
        
        # Risk Factors
        risk_factors = report.get("risk_factors", [])
        lines.append("## Risk Factors")
        lines.append("")
        if risk_factors:
            for rf in risk_factors:
                lines.append(f"- {rf}")
        else:
            lines.append("*None identified*")
        lines.append("")
        
        # Visual Findings
        visual = report.get("visual_findings")
        if visual:
            lines.append("## Visual Findings")
            lines.append("")
            lines.append(visual)
            lines.append("")
        
        # Suggested Differential - Table format
        differential = report.get("suggested_differential", [])
        lines.append("## Suggested Differential")
        lines.append("")
        if differential:
            lines.append("| Condition | Likelihood | Source |")
            lines.append("|-----------|------------|--------|")
            for d in differential:
                cond = d.get("condition", "Unknown")
                if isinstance(cond, list):
                    cond = ", ".join(str(c) for c in cond[:3])  # Limit to 3
                like = d.get("likelihood", "Unknown")
                source = d.get("source", "Analysis")
                lines.append(f"| {cond} | {like} | {source} |")
        else:
            lines.append("*No differential diagnosis available*")
        lines.append("")
        
        # Recommended Next Steps
        steps = report.get("recommended_next_steps", [])
        lines.append("## Recommended Next Steps")
        lines.append("")
        if steps:
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
        else:
            lines.append("*Follow up with healthcare provider*")
        lines.append("")
        
        # Disclaimer
        lines.append("---")
        lines.append("")
        lines.append("### Disclaimer")
        lines.append("")
        disclaimer = report.get("disclaimer", "")
        lines.append(f"*{disclaimer}*")
        
        return "\n".join(lines)


# Singleton instance
_clinical_report_generator = None

def get_clinical_report_generator() -> ClinicalReportGenerator:
    """Get or create the singleton ClinicalReportGenerator instance."""
    global _clinical_report_generator
    if _clinical_report_generator is None:
        _clinical_report_generator = ClinicalReportGenerator()
    return _clinical_report_generator
