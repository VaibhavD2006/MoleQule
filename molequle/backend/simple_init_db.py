#!/usr/bin/env python3
"""
Simple database initialization script for MoleQule backend.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.models.database import create_tables

def main():
    """Initialize the database"""
    print("Creating MoleQule database tables...")
    
    try:
        # Create all tables
        create_tables()
        print("✓ Database tables created successfully!")
        print("✓ You can now start the backend with: python main.py")
        
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 