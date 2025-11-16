#!/usr/bin/env python
"""Simple test script to debug the backend"""
import sys
sys.path.insert(0, 'c:\\Users\\HP\\Downloads\\gen-ai_proj')

try:
    print("Testing imports...")
    from backend.risk_model import RiskModel
    print("✓ RiskModel imported")
    
    from backend.storyteller import StoryTeller
    print("✓ StoryTeller imported")
    
    print("\nTesting RiskModel...")
    risk_model = RiskModel("models/risk_lstm.h5")
    print("✓ RiskModel initialized")
    
    print("\nTesting StoryTeller...")
    story_model = StoryTeller("models/llama-3-8b.gguf")
    print("✓ StoryTeller initialized")
    
    print("\nTesting prediction...")
    import numpy as np
    angles = np.random.rand(30, 3) * 180
    risk = risk_model.predict_risk(angles)
    print(f"✓ Risk prediction: {risk:.2f}%")
    
    print("\nTesting explanation...")
    hip, knee, shoulder = angles[-1]
    explanation = story_model.explain(hip, knee, shoulder, risk)
    print(f"✓ Explanation generated: {explanation[:100]}...")
    
    print("\n✓ All tests passed!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
