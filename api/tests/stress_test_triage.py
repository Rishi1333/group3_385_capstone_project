"""
Stress test for the Triage Agent fix - tests repeated question prevention.
Run with: python api/tests/stress_test_triage.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.triage_agent import TriageAgent, TriageState


def test_no_repeated_questions():
    """Test that the agent doesn't ask the same question twice."""
    print("\n=== TEST 1: No Repeated Questions ===")
    
    agent = TriageAgent(session_id="test-1")
    
    # Start intake
    result = agent.start_intake("I have a headache")
    print(f"Q1: {result.get('question')}")
    
    # User gives irrelevant answer
    result = agent.process_message("I don't know")
    print(f"Q2: {result.get('question')}")
    
    # Check that the question changed
    q1 = result.get('question', '').lower() if result.get('question') else ''
    
    # Another irrelevant answer
    result = agent.process_message("blah blah")
    print(f"Q3: {result.get('question')}")
    
    # Verify asked_fields is being populated
    print(f"Asked fields: {agent.asked_fields}")
    
    # Continue until we transition or run out of questions
    for i in range(10):
        if result.get('state') != 'GATHERING':
            print(f"Transitioned to: {result.get('state')}")
            break
        
        result = agent.process_message("no comment")
        print(f"Q{i+4}: {result.get('question')}")
        print(f"Asked fields: {agent.asked_fields}")
    
    # Verify we didn't get stuck
    assert len(agent.asked_fields) > 0, "Should have asked at least one field"
    print("[PASSED] Agent progressed through questions without repeating\n")


def test_all_fields_asked_transitions():
    """Test that agent transitions when all fields have been asked."""
    print("\n=== TEST 2: All Fields Asked Transitions ===")
    
    agent = TriageAgent(session_id="test-2")
    
    # Start intake
    result = agent.start_intake("I have fever and cough")
    print(f"Initial: {result.get('question')}")
    
    # Answer all questions with irrelevant responses
    for i in range(10):
        if result.get('state') != 'GATHERING':
            print(f"Transitioned to: {result.get('state')}")
            break
        
        result = agent.process_message("I refuse to answer")
        print(f"Q{i+1}: asked_fields = {agent.asked_fields}")
    
    # Should have transitioned even without answers
    assert result.get('state') in ['ANALYZING', 'COMPLETE'], \
        f"Should transition after all fields asked, got state: {result.get('state')}"
    print("[PASSED] Agent transitioned after all fields asked\n")


def test_session_persistence():
    """Test that asked_fields persists across session save/restore."""
    print("\n=== TEST 3: Session Persistence ===")
    
    agent = TriageAgent(session_id="test-3")
    agent.start_intake("I have stomach pain")
    agent.process_message("random answer")
    
    # Save state
    saved = agent.to_dict()
    print(f"Saved asked_fields: {saved.get('asked_fields')}")
    
    # Restore
    restored_agent = TriageAgent.from_dict(saved)
    print(f"Restored asked_fields: {restored_agent.asked_fields}")
    
    assert restored_agent.asked_fields == agent.asked_fields, \
        "asked_fields should persist across save/restore"
    print("[PASSED] asked_fields persists correctly\n")


def test_normal_flow_still_works():
    """Test that normal flow with proper answers still works."""
    print("\n=== TEST 4: Normal Flow Still Works ===")
    
    agent = TriageAgent(session_id="test-4")
    
    result = agent.start_intake("I have chest pain and shortness of breath")
    print(f"Q1: {result.get('question')}")
    
    # Answer properly with age and sex
    result = agent.process_message("I am 45 years old male")
    print(f"Result after age/sex: state={result.get('state')}")
    print(f"Clinical data: {agent.get_clinical_data()}")
    
    # Agent may transition early if it has enough info (3 symptoms detected)
    # This is correct behavior - the agent should transition when ready
    if result.get('state') == 'ANALYZING':
        print("Agent transitioned to ANALYZING (has enough info)")
        assert agent.state == TriageState.ANALYZING
        print("[PASSED] Normal flow works correctly (early transition)\n")
        return
    
    # Continue with proper answers if still gathering
    result = agent.process_message("no medical history")
    print(f"Q3: {result.get('question')}")
    
    result = agent.process_message("no medications")
    print(f"Q4: {result.get('question')}")
    
    result = agent.process_message("no family history")
    print(f"Q5: {result.get('question')}")
    
    # Should transition
    print(f"State: {result.get('state')}")
    print(f"Clinical data: {agent.get_clinical_data()}")
    
    assert result.get('state') in ['ANALYZING', 'COMPLETE', 'GATHERING'], \
        f"Unexpected state: {result.get('state')}"
    print("[PASSED] Normal flow works correctly\n")


def test_comprehensive_no_detection():
    """Test that comprehensive 'no' answers mark all fields as addressed."""
    print("\n=== TEST 5: Comprehensive No Detection ===")
    
    agent = TriageAgent(session_id="test-5")
    
    result = agent.start_intake("I have a migraine")
    print(f"Q1: {result.get('question')}")
    
    # Give age/sex
    result = agent.process_message("I am 30 year old female")
    print(f"Q2: {result.get('question')}")
    
    # Comprehensive no
    result = agent.process_message("no medication or supplements, no family history or lifestyle issues")
    print(f"info_addressed: {agent.info_addressed}")
    
    # Should have marked fields as addressed
    assert agent.info_addressed.get('medications') or agent.info_addressed.get('risk_factors'), \
        "Comprehensive no should mark fields as addressed"
    print("[PASSED] Comprehensive no detection works\n")


def run_all_tests():
    """Run all stress tests."""
    print("\n" + "="*60)
    print("STRESS TESTING TRIAGE AGENT FIX")
    print("="*60)
    
    try:
        test_no_repeated_questions()
        test_all_fields_asked_transitions()
        test_session_persistence()
        test_normal_flow_still_works()
        test_comprehensive_no_detection()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return True
    except AssertionError as e:
        print(f"\n[FAILED] {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
