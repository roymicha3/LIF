#!/usr/bin/env python3
"""
Check what accuracy metrics are actually stored in the database.
"""

import sqlite3
import pandas as pd

def check_database_metrics():
    """Check what metrics are actually stored in the database."""
    
    db_path = r'outputs\sequential_workspace\artifacts\experiment.db'
    
    print(f"Checking database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check all tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in database: {[table[0] for table in tables]}")
        
        # Check METRIC table
        print("\n=== METRIC TABLE ===")
        metric_df = pd.read_sql_query("SELECT * FROM METRIC", conn)
        print(f"Metric table columns: {metric_df.columns.tolist()}")
        print(f"Metric table shape: {metric_df.shape}")
        print("All metrics in database:")
        for _, row in metric_df.iterrows():
            print(f"  - {row['type']} (ID: {row['id']})")
        
        # Check for accuracy metrics in METRIC table
        print("\n=== SEARCHING FOR ACCURACY METRICS IN METRIC TABLE ===")
        accuracy_metrics = metric_df[metric_df['type'].str.contains('acc', case=False, na=False)]
        if len(accuracy_metrics) > 0:
            print("Found accuracy metrics in METRIC table:")
            for _, row in accuracy_metrics.iterrows():
                print(f"  - {row['type']} (ID: {row['id']})")
        else:
            print("No accuracy metrics found in METRIC table")
        
        # Check for validation metrics in METRIC table
        print("\n=== SEARCHING FOR VALIDATION METRICS IN METRIC TABLE ===")
        validation_metrics = metric_df[metric_df['type'].str.contains('val', case=False, na=False)]
        if len(validation_metrics) > 0:
            print("Found validation metrics in METRIC table:")
            for _, row in validation_metrics.iterrows():
                print(f"  - {row['type']} (ID: {row['id']})")
        else:
            print("No validation metrics found in METRIC table")
        
        # Check BATCH_METRIC table
        print("\n=== BATCH_METRIC TABLE ===")
        batch_metric_df = pd.read_sql_query("SELECT * FROM BATCH_METRIC LIMIT 10", conn)
        print(f"Batch_metric table columns: {batch_metric_df.columns.tolist()}")
        print(f"Batch_metric table shape: {batch_metric_df.shape}")
        print("Sample batch metrics:")
        print(batch_metric_df.head())
        
        # Check EPOCH table
        print("\n=== EPOCH TABLE ===")
        epoch_df = pd.read_sql_query("SELECT * FROM EPOCH LIMIT 10", conn)
        print(f"Epoch table columns: {epoch_df.columns.tolist()}")
        print(f"Epoch table shape: {epoch_df.shape}")
        print("Sample epoch data:")
        print(epoch_df.head())
        
        # Check EPOCH_METRIC table
        print("\n=== EPOCH_METRIC TABLE ===")
        epoch_metric_df = pd.read_sql_query("SELECT * FROM EPOCH_METRIC LIMIT 10", conn)
        print(f"Epoch_metric table columns: {epoch_metric_df.columns.tolist()}")
        print(f"Epoch_metric table shape: {epoch_metric_df.shape}")
        print("Sample epoch metrics:")
        print(epoch_metric_df.head())
        
        # Check RESULTS_METRIC table
        print("\n=== RESULTS_METRIC TABLE ===")
        results_metric_df = pd.read_sql_query("SELECT * FROM RESULTS_METRIC LIMIT 10", conn)
        print(f"Results_metric table columns: {results_metric_df.columns.tolist()}")
        print(f"Results_metric table shape: {results_metric_df.shape}")
        print("Sample results metrics:")
        print(results_metric_df.head())
        
        # Check for accuracy data in RESULTS_METRIC
        print("\n=== CHECKING FOR ACCURACY DATA IN RESULTS_METRIC ===")
        acc_data = pd.read_sql_query("""
            SELECT rm.metric_id, m.type, rm.value, r.trial_run_id
            FROM RESULTS_METRIC rm
            JOIN METRIC m ON rm.metric_id = m.id
            JOIN RESULTS r ON rm.results_id = r.id
            WHERE m.type LIKE '%acc%'
            ORDER BY m.type, r.trial_run_id
            LIMIT 10
        """, conn)
        
        if len(acc_data) > 0:
            print("Found accuracy data in RESULTS_METRIC:")
            print(acc_data)
        else:
            print("No accuracy data found in RESULTS_METRIC")
        
        conn.close()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database_metrics()
