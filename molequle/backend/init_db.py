#!/usr/bin/env python3
"""
Database initialization script for MoleQule backend.
Run this script to create all database tables.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.models.database import create_tables, engine
from sqlalchemy import text

def main():
    """Initialize the database"""
    print("Initializing MoleQule database...")
    
    try:
        # Create all tables
        create_tables()
        
        # Test the connection and list tables
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
            tables = [row[0] for row in result]
            
        print("✓ Database initialized successfully!")
        print(f"✓ Created tables: {', '.join(tables)}")
        print(f"✓ Database file: {os.path.abspath('molequle.db')}")
        
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 