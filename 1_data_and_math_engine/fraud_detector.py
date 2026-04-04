import pandas as pd
import networkx as nx
import json
from datetime import datetime, timedelta
from loguru import logger

class FraudDetector:
    def __init__(self, csv_path="./mock_data/upi_logs.csv"):
        self.csv_path = csv_path
        self.df = None
        self.G = nx.DiGraph()

    def load_and_filter(self, time_window_hours=48):
        """Loads logs and filters for the most recent window."""
        self.df = pd.read_csv(self.csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Filter for recent transactions to keep the graph efficient
        latest_time = self.df['timestamp'].max()
        cutoff = latest_time - timedelta(hours=time_window_hours)
        recent_df = self.df[self.df['timestamp'] >= cutoff]
        
        # Build Directed Graph: Edges weighted by transaction amount
        self.G = nx.from_pandas_edgelist(
            recent_df, 
            source='sender_id', 
            target='receiver_id', 
            edge_attr='amount', 
            create_using=nx.DiGraph()
        )
        logger.info(f"Graph built with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")

    def detect_circular_loops(self, min_len=3, max_len=4):
        """Finds simple cycles of specific lengths (e.g., A->B->C->A)."""
        alerts = []
        
        # nx.simple_cycles uses Johnson's algorithm (efficient for sparse graphs)
        cycles = list(nx.simple_cycles(self.G))
        
        for cycle in cycles:
            if min_len <= len(cycle) <= max_len:
                # Calculate total volume within this specific loop
                total_volume = 0
                edges_involved = []
                
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)] # Wraps back to start
                    
                    amount = self.G[u][v]['amount']
                    total_volume += amount
                    edges_involved.append(f"{u} -> {v}")

                alert_payload = {
                    "alert_type": "CIRCULAR_TRANSACTION_FRAUD",
                    "severity": "HIGH",
                    "nodes_involved": cycle,
                    "hop_count": len(cycle),
                    "total_loop_volume": round(total_volume, 2),
                    "path": " -> ".join(cycle + [cycle[0]])
                }
                alerts.append(alert_payload)
        
        return alerts

    def get_fraud_report(self):
        """Returns the final JSON for the Orchestrator/Frontend."""
        self.load_and_filter()
        detected_cycles = self.detect_circular_loops()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(detected_cycles),
            "alerts": detected_cycles
        }
        return report

if __name__ == "__main__":
    detector = FraudDetector()
    report = detector.get_fraud_report()
    
    # Output formatted JSON for the terminal
    print(json.dumps(report, indent=4))