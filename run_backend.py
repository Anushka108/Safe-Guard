#!/usr/bin/env python
"""Test server with better error handling"""
import sys
import traceback

if __name__ == "__main__":
    try:
        print("Starting backend...")
        from backend.main import app
        import uvicorn
        
        print("Routes available:")
        for route in app.routes:
            print(f"  - {route.path}")
        
        print("\nStarting uvicorn server...")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
        
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
