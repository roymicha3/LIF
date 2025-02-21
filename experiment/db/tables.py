from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'EXPERIMENT'
    id = Column(String(255), primary_key=True)
    title = Column(String(255))
    desc = Column(Text)
    start_time = Column(DateTime)
    update_time = Column(DateTime)
    config = Column(Text)
    dataset_id = Column(String(255), ForeignKey('DATASET.id'))
    dataset = relationship("Dataset")
    trials = relationship("Trial", back_populates="experiment")

class Trial(Base):
    __tablename__ = 'TRIAL'
    id = Column(String(255), primary_key=True)
    name = Column(String(255))
    desc = Column(Text)
    start_time = Column(DateTime)
    update_time = Column(DateTime)
    config = Column(Text)
    experiment_id = Column(String(255), ForeignKey('EXPERIMENT.id'))
    experiment = relationship("Experiment", back_populates="trials")
    trial_runs = relationship("TrialRun", back_populates="trial")

class TrialRun(Base):
    __tablename__ = 'TRIAL_RUN'
    id = Column(String(255), primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    status = Column(String(255))
    trial_id = Column(String(255), ForeignKey('TRIAL.id'))
    trial = relationship("Trial", back_populates="trial_runs")
    results = relationship("Results", uselist=False, back_populates="trial_run")
    epochs = relationship("Epoch", back_populates="trial_run")
    artifacts = relationship("Artifact", back_populates="trial_run")
    logs = relationship("Logs", back_populates="trial_run")

class Results(Base):
    __tablename__ = 'RESULTS'
    id = Column(String(255), primary_key=True)
    trial_run_id = Column(String(255), ForeignKey('TRIAL_RUN.id'))
    total_accuracy = Column(Float)
    accuracy_per_label = Column(Text)
    total_loss = Column(Float)
    loss_per_label = Column(Text)
    trial_run = relationship("TrialRun", back_populates="results")
    artifacts = relationship("Artifact", back_populates="results")

class Epoch(Base):
    __tablename__ = 'EPOCH'
    id = Column(String(255), primary_key=True)
    trial_run_id = Column(String(255), ForeignKey('TRIAL_RUN.id'))
    index = Column(Integer)
    total_accuracy = Column(Float)
    accuracy_per_label = Column(Text)
    total_loss = Column(Float)
    loss_per_label = Column(Text)
    trial_run = relationship("TrialRun", back_populates="epochs")
    artifacts = relationship("Artifact", back_populates="epoch")

class Dataset(Base):
    __tablename__ = 'DATASET'
    id = Column(String(255), primary_key=True)
    size = Column(Float)
    location = Column(String(255))
    config = Column(Text)
    experiments = relationship("Experiment", back_populates="dataset")
    encoders = relationship("Encoder", secondary="ENCODER_DATASET")

class Encoder(Base):
    __tablename__ = 'ENCODER'
    type = Column(String(255), primary_key=True)
    config = Column(Text)
    datasets = relationship("Dataset", secondary="ENCODER_DATASET")

class Artifact(Base):
    __tablename__ = 'ARTIFACT'
    id = Column(String(255), primary_key=True)
    type = Column(String(255))
    config = Column(Text)
    location = Column(String(255))
    trial_run_id = Column(String(255), ForeignKey('TRIAL_RUN.id'))
    epoch_id = Column(String(255), ForeignKey('EPOCH.id'))
    results_id = Column(String(255), ForeignKey('RESULTS.id'))
    trial_run = relationship("TrialRun", back_populates="artifacts")
    epoch = relationship("Epoch", back_populates="artifacts")
    results = relationship("Results", back_populates="artifacts")

class Logs(Base):
    __tablename__ = 'LOGS'
    id = Column(String(255), primary_key=True)
    location = Column(String(255))
    trial_run_id = Column(String(255), ForeignKey('TRIAL_RUN.id'))
    trial_run = relationship("TrialRun", back_populates="logs")

class ExperimentDataset(Base):
    __tablename__ = 'EXPERIMENT_DATASET'
    experiment_id = Column(String(255), ForeignKey('EXPERIMENT.id'), primary_key=True)
    dataset_id = Column(String(255), ForeignKey('DATASET.id'), primary_key=True)

class EncoderDataset(Base):
    __tablename__ = 'ENCODER_DATASET'
    encoder_type = Column(String(255), ForeignKey('ENCODER.type'), primary_key=True)
    dataset_id = Column(String(255), ForeignKey('DATASET.id'), primary_key=True)
    
