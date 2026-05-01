from sqlalchemy import (create_engine, Column, Integer,
                        String, Float, DateTime, Text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./neurovoice.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(100))
    age        = Column(Integer)
    condition  = Column(String(100))
    ward       = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    __tablename__ = "sessions"
    id            = Column(Integer, primary_key=True, index=True)
    patient_id    = Column(Integer)
    started_at    = Column(DateTime, default=datetime.utcnow)
    ended_at      = Column(DateTime, nullable=True)
    total_decoded = Column(Integer, default=0)


class DecodedMessage(Base):
    __tablename__ = "decoded_messages"
    id               = Column(Integer, primary_key=True, index=True)
    session_id       = Column(Integer)
    patient_id       = Column(Integer)
    word             = Column(String(50))
    confidence       = Column(Float)
    timestamp        = Column(DateTime, default=datetime.utcnow)
    all_probabilities = Column(Text)


# This creates all tables automatically when file runs
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()