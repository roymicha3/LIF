from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, JSON, Table, ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

# Association tables for many-to-many relationships
results_metric = Table('results_metric', Base.metadata,
    Column('results_id', Integer, ForeignKey('results.trial_run_id')),
    Column('metric_id', Integer, ForeignKey('metric.id'))
)

results_artifact = Table('results_artifact', Base.metadata,
    Column('results_id', Integer, ForeignKey('results.trial_run_id')),
    Column('artifact_id', Integer, ForeignKey('artifact.id'))
)

epoch_metric = Table('epoch_metric', Base.metadata,
    Column('epoch_idx', Integer, primary_key=True),
    Column('epoch_trial_run_id', Integer, primary_key=True),
    Column('metric_id', Integer, ForeignKey('metric.id'), primary_key=True),
    ForeignKeyConstraint(['epoch_idx', 'epoch_trial_run_id'], ['epoch.idx', 'epoch.trial_run_id'])
)

epoch_artifact = Table('epoch_artifact', Base.metadata,
    Column('epoch_idx', Integer, primary_key=True),
    Column('epoch_trial_run_id', Integer, primary_key=True),
    Column('artifact_id', Integer, ForeignKey('artifact.id'), primary_key=True),
    ForeignKeyConstraint(['epoch_idx', 'epoch_trial_run_id'], ['epoch.idx', 'epoch.trial_run_id'])
)

trial_run_artifact = Table('trial_run_artifact', Base.metadata,
    Column('trial_run_id', Integer, ForeignKey('trial_run.id')),
    Column('artifact_id', Integer, ForeignKey('artifact.id'))
)

class Experiment(Base):
    __tablename__ = 'experiment'
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    desc = Column(String)
    start_time = Column(DateTime, nullable=False)
    update_time = Column(DateTime, nullable=False)
    trials = relationship("Trial", back_populates="experiment")

class Trial(Base):
    __tablename__ = 'trial'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    experiment_id = Column(Integer, ForeignKey('experiment.id'), nullable=False)
    start_time = Column(DateTime, nullable=False)
    update_time = Column(DateTime, nullable=False)
    experiment = relationship("Experiment", back_populates="trials")
    trial_runs = relationship("TrialRun", back_populates="trial")

class TrialRun(Base):
    __tablename__ = 'trial_run'
    id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trial.id'), nullable=False)
    status = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False)
    update_time = Column(DateTime, nullable=False)
    trial = relationship("Trial", back_populates="trial_runs")
    results = relationship("Results", back_populates="trial_run", uselist=False)
    epochs = relationship("Epoch", back_populates="trial_run")
    artifacts = relationship("Artifact", secondary=trial_run_artifact, back_populates="trial_runs")

class Results(Base):
    __tablename__ = 'results'
    trial_run_id = Column(Integer, ForeignKey('trial_run.id'), primary_key=True)
    time = Column(DateTime, nullable=False)
    trial_run = relationship("TrialRun", back_populates="results")
    metrics = relationship("Metric", secondary=results_metric, back_populates="results")
    artifacts = relationship("Artifact", secondary=results_artifact, back_populates="results")

class Epoch(Base):
    __tablename__ = 'epoch'
    idx = Column(Integer, primary_key=True)
    trial_run_id = Column(Integer, ForeignKey('trial_run.id'), primary_key=True)
    time = Column(DateTime, nullable=False)
    trial_run = relationship("TrialRun", back_populates="epochs")
    metrics = relationship("Metric", secondary=epoch_metric, back_populates="epochs")
    artifacts = relationship("Artifact", secondary=epoch_artifact, back_populates="epochs")

class Metric(Base):
    __tablename__ = 'metric'
    id = Column(Integer, primary_key=True)
    type = Column(String(50), nullable=False)
    total_val = Column(Float, nullable=False)
    per_label_val = Column(JSON)
    results = relationship("Results", secondary=results_metric, back_populates="metrics")
    epochs = relationship("Epoch", secondary=epoch_metric, back_populates="metrics")

class Artifact(Base):
    __tablename__ = 'artifact'
    id = Column(Integer, primary_key=True)
    type = Column(String(50), nullable=False)
    loc = Column(String(255), nullable=False)
    results = relationship("Results", secondary=results_artifact, back_populates="artifacts")
    epochs = relationship("Epoch", secondary=epoch_artifact, back_populates="artifacts")
    trial_runs = relationship("TrialRun", secondary=trial_run_artifact, back_populates="artifacts")



