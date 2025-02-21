from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiment'
    id = Column(String, primary_key=True)
    title = Column(String)
    desc = Column(Text)
    start_time = Column(DateTime)
    update_time = Column(DateTime)
    config = Column(Text)
    dataset_id = Column(String, ForeignKey('dataset.id'))

    trials = relationship("Trial", back_populates="experiment")

class Trial(Base):
    __tablename__ = 'trial'
    id = Column(String, primary_key=True)
    name = Column(String)
    desc = Column(Text)
    start_time = Column(DateTime)
    update_time = Column(DateTime)
    config = Column(Text)
    experiment_id = Column(String, ForeignKey('experiment.id'))

    experiment = relationship("Experiment", back_populates="trials")
    trial_runs = relationship("TrialRun", back_populates="trial")

class TrialRun(Base):
    __tablename__ = 'trial_run'
    id = Column(String, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    status = Column(String)
    trial_id = Column(String, ForeignKey('trial.id'))

    trial = relationship("Trial", back_populates="trial_runs")
    results = relationship("Results", back_populates="trial_run")
    epochs = relationship("Epoch", back_populates="trial_run")
    logs = relationship("Logs", back_populates="trial_run")
    artifacts = relationship("Artifact", back_populates="trial_run")

class Results(Base):
    __tablename__ = 'results'
    id = Column(String, primary_key=True)
    trial_run_id = Column(String, ForeignKey('trial_run.id'))
    total_accuracy = Column(Float)
    accuracy_per_label = Column(Text)
    total_loss = Column(Float)
    loss_per_label = Column(Text)

    trial_run = relationship("TrialRun", back_populates="results")
    artifacts = relationship("Artifact", back_populates="results")

class Epoch(Base):
    __tablename__ = 'epoch'
    id = Column(String, primary_key=True)
    trial_run_id = Column(String, ForeignKey('trial_run.id'))
    index = Column(Integer)
    total_accuracy = Column(Float)
    accuracy_per_label = Column(Text)
    total_loss = Column(Float)
    loss_per_label = Column(Text)

    trial_run = relationship("TrialRun", back_populates="epochs")
    artifacts = relationship("Artifact", back_populates="epoch")

class Dataset(Base):
    __tablename__ = 'dataset'
    id = Column(String, primary_key=True)
    size = Column(Float)
    location = Column(String)
    config = Column(Text)

class Encoder(Base):
    __tablename__ = 'encoder'
    type = Column(String, primary_key=True)
    config = Column(Text)

class Artifact(Base):
    __tablename__ = 'artifact'
    id = Column(String, primary_key=True)
    type = Column(String)
    config = Column(Text)
    location = Column(String)
    trial_run_id = Column(String, ForeignKey('trial_run.id'))
    epoch_id = Column(String, ForeignKey('epoch.id'))
    results_id = Column(String, ForeignKey('results.id'))

    trial_run = relationship("TrialRun", back_populates="artifacts")
    epoch = relationship("Epoch", back_populates="artifacts")
    results = relationship("Results", back_populates="artifacts")

class Logs(Base):
    __tablename__ = 'logs'
    id = Column(String, primary_key=True)
    location = Column(String)
    trial_run_id = Column(String, ForeignKey('trial_run.id'))

    trial_run = relationship("TrialRun", back_populates="logs")

# Database Initialization
def init_db(uri='sqlite:///experiment_tracking.db'):
    engine = create_engine(uri, echo=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

Session = init_db()
