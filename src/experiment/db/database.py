import os
from functools import wraps

from experiment.db.db_manager import DatabaseManager


class SingletonDatabaseManager:
    _instance = None

    def __init__(self):
        raise RuntimeError('Call initialize() instead')

    @classmethod
    def initialize(cls, db_path):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.db_manager = DatabaseManager(db_path)
            if not os.path.exists(db_path):
                cls._instance.db_manager.create_tables()
        
        return cls._instance

    @classmethod
    def instance(cls):
        if cls._instance is None:
            raise RuntimeError("DatabaseManager instance not initialized. Call SingletonDatabaseManager.initialize(db_path) first.")
        return cls._instance

    def __getattr__(self, name):
        return getattr(self.db_manager, name)

def singleton_db_manager(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if SingletonDatabaseManager._instance is None:
            raise RuntimeError("DatabaseManager instance not initialized. Call SingletonDatabaseManager.initialize(db_path) first.")
        return func(*args, **kwargs)
    return wrapper

# Decorate all methods of DatabaseManager with singleton_db_manager
for attr_name in dir(DatabaseManager):
    attr = getattr(DatabaseManager, attr_name)
    if callable(attr) and not attr_name.startswith("__"):
        setattr(SingletonDatabaseManager, attr_name, singleton_db_manager(attr))


DB = SingletonDatabaseManager
