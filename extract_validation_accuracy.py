#!/usr/bin/env python3
"""
Direct extraction of validation accuracy from the database.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def extract_validation_accuracy_direct():
    """Extract validation accuracy directly from the database using SQL."""
    
    db_path = r'outputs\sequential_workspace\artifacts\experiment.db'
    
    print(f"Directly extracting validation accuracy from: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # First, let's see what's in the METRIC table for validation accuracy
        print("\n=== CHECKING METRIC TABLE FOR VALIDATION ACCURACY ===")
        
        # Get all validation accuracy metrics
        val_acc_metrics = pd.read_sql_query("""
            SELECT id, type, total_val, per_label_val
            FROM METRIC 
            WHERE type = 'val_acc'
            LIMIT 10
        """, conn)
        
        print(f"Found {len(val_acc_metrics)} validation accuracy metrics in METRIC table")
        print(val_acc_metrics.head())
        
        # Check if there are any epoch-level metrics stored
        print("\n=== CHECKING EPOCH_METRIC TABLE ===")
        
        epoch_metrics = pd.read_sql_query("""
            SELECT em.epoch_idx, em.epoch_trial_run_id, em.metric_id, m.type
            FROM EPOCH_METRIC em
            JOIN METRIC m ON em.metric_id = m.id
            WHERE m.type = 'val_acc'
            LIMIT 10
        """, conn)
        
        print(f"Found {len(epoch_metrics)} validation accuracy records in EPOCH_METRIC")
        if len(epoch_metrics) > 0:
            print(epoch_metrics.head())
        
        # Check if validation accuracy is stored in BATCH_METRIC
        print("\n=== CHECKING BATCH_METRIC TABLE ===")
        
        batch_val_acc = pd.read_sql_query("""
            SELECT bm.batch_idx, bm.epoch_idx, bm.trial_run_id, bm.metric_id, m.type
            FROM BATCH_METRIC bm
            JOIN METRIC m ON bm.metric_id = m.id
            WHERE m.type = 'val_acc'
            LIMIT 10
        """, conn)
        
        print(f"Found {len(batch_val_acc)} validation accuracy records in BATCH_METRIC")
        if len(batch_val_acc) > 0:
            print(batch_val_acc.head())
        
        # Check if validation accuracy is stored in RESULTS_METRIC
        print("\n=== CHECKING RESULTS_METRIC TABLE ===")
        
        results_val_acc = pd.read_sql_query("""
            SELECT rm.results_id, rm.metric_id, m.type, r.trial_run_id
            FROM RESULTS_METRIC rm
            JOIN METRIC m ON rm.metric_id = m.id
            JOIN RESULTS r ON rm.results_id = r.id
            WHERE m.type = 'val_acc'
            LIMIT 10
        """, conn)
        
        print(f"Found {len(results_val_acc)} validation accuracy records in RESULTS_METRIC")
        if len(results_val_acc) > 0:
            print(results_val_acc.head())
        
        # Let's try to find where the validation accuracy values are actually stored
        print("\n=== SEARCHING FOR VALIDATION ACCURACY VALUES ===")
        
        # Check if there's a separate table for metric values
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"All tables: {[table[0] for table in tables]}")
        
        # Look for any table that might contain metric values
        for table_name in [table[0] for table in tables]:
            if 'value' in table_name.lower() or 'metric' in table_name.lower():
                print(f"\nChecking table: {table_name}")
                try:
                    table_info = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5", conn)
                    print(f"Columns: {table_info.columns.tolist()}")
                    print(f"Shape: {table_info.shape}")
                    if len(table_info) > 0:
                        print(table_info.head())
                except Exception as e:
                    print(f"Error reading table {table_name}: {e}")
        
        # Let's check if the METRIC table has the values directly
        print("\n=== CHECKING METRIC TABLE FOR VALUES ===")
        
        # Get a sample of validation accuracy metrics with their values
        val_acc_with_values = pd.read_sql_query("""
            SELECT id, type, total_val, per_label_val
            FROM METRIC 
            WHERE type = 'val_acc'
            ORDER BY id
            LIMIT 20
        """, conn)
        
        print("Validation accuracy metrics with values:")
        print(val_acc_with_values)
        
        # Check if total_val contains the actual accuracy values
        if len(val_acc_with_values) > 0:
            print(f"\nSample validation accuracy values:")
            for _, row in val_acc_with_values.iterrows():
                print(f"  ID {row['id']}: total_val = {row['total_val']}, per_label_val = {row['per_label_val']}")
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_validation_accuracy_direct()





