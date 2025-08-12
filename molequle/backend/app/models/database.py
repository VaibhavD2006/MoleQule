from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./molequle.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class
Base = declarative_base()

class Job(Base):
    """Job model for tracking molecular processing tasks"""
    __tablename__ = "jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=True)  # For future auth
    job_id = Column(String, unique=True, index=True)
    input_file = Column(String)  # File path or S3 URL
    input_format = Column(String)  # SMILES, MOL, XYZ
    original_filename = Column(String)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    result_file = Column(String, nullable=True)  # S3 path or local path
    error_message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # in seconds
    
    def __repr__(self):
        return f"<Job(job_id='{self.job_id}', status='{self.status}')>"

class Analog(Base):
    """Analog model for storing generated molecular analogs"""
    __tablename__ = "analogs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, index=True)
    analog_id = Column(String)
    smiles = Column(String)
    binding_affinity = Column(Float)
    final_score = Column(Float)
    rank = Column(Integer)
    energy = Column(Float, nullable=True)
    homo_lumo_gap = Column(Float, nullable=True)
    dipole_moment = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Analog(analog_id='{self.analog_id}', rank={self.rank})>"

class User(Base):
    """User model for future authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User(email='{self.email}')>"

# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

# Database dependency
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 