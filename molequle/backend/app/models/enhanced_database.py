#!/usr/bin/env python3
"""
Enhanced Database Models for MoleQule
Comprehensive drug property storage
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import json

Base = declarative_base()

class EnhancedJob(Base):
    """Enhanced job model for comprehensive analysis."""
    __tablename__ = "enhanced_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    input_file_path = Column(String, nullable=False)
    comprehensive_analysis = Column(Boolean, default=True)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    processing_time = Column(Float)
    analysis_type = Column(String, default="comprehensive")  # basic, comprehensive
    summary_json = Column(JSON)  # Store comprehensive summary as JSON

class EnhancedAnalog(Base):
    """Enhanced analog model with comprehensive properties."""
    __tablename__ = "enhanced_analogs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, index=True, nullable=False)
    analog_id = Column(String, nullable=False)
    rank = Column(Integer)
    smiles = Column(String, nullable=False)
    
    # Basic properties
    binding_affinity = Column(Float)
    energy = Column(Float)
    homo_lumo_gap = Column(Float)
    final_score = Column(Float)
    
    # Enhanced properties
    comprehensive_score = Column(Float)
    admet_score = Column(Float)
    cytotoxicity_score = Column(Float)
    cancer_pathway_score = Column(Float)
    safety_score = Column(Float)
    
    # Clinical assessment
    clinical_readiness = Column(String)  # ready, needs_optimization, requires_work
    development_priority = Column(String)  # high, medium, low
    risk_level = Column(String)  # low, moderate, high, very_high
    
    # Detailed analysis (stored as JSON)
    experimental_validation_json = Column(JSON)
    cytotoxicity_predictions_json = Column(JSON)
    admet_predictions_json = Column(JSON)
    cancer_pathway_analysis_json = Column(JSON)
    comprehensive_scoring_json = Column(JSON)
    detailed_analysis_json = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AnalysisCapability(Base):
    """Model to track analysis capabilities and their status."""
    __tablename__ = "analysis_capabilities"
    
    id = Column(Integer, primary_key=True, index=True)
    capability_name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    status = Column(String, default="active")  # active, inactive, error
    version = Column(String)
    last_updated = Column(DateTime, default=datetime.utcnow)
    configuration_json = Column(JSON)

class ValidationDataset(Base):
    """Model to track experimental validation datasets."""
    __tablename__ = "validation_datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, nullable=False)
    source = Column(String, nullable=False)  # PDBbind, ChEMBL, BindingDB
    description = Column(Text)
    compound_count = Column(Integer)
    last_updated = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)

class ClinicalAssessment(Base):
    """Model to track clinical assessments and recommendations."""
    __tablename__ = "clinical_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    analog_id = Column(String, index=True, nullable=False)
    assessment_type = Column(String, nullable=False)  # safety, efficacy, admet, pathway
    assessment_score = Column(Float)
    assessment_classification = Column(String)
    recommendations_json = Column(JSON)
    risk_factors_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 