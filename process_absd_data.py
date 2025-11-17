"""
Process ABSD (Airport Surface Movement) data for turnaround prediction
"""

import sys
import os

# Add the Processed ABSD data directory to path
absd_dir = os.path.join(os.path.dirname(__file__), 'Processed ABSD data')
sys.path.insert(0, absd_dir)

try:
    from gm_parse import read_gm_dict, gm_parser2
except ImportError:
    # Try alternative import
    import importlib.util
    gm_parse_path = os.path.join(absd_dir, 'gm_parse (1).py')
    spec = importlib.util.spec_from_file_location("gm_parse", gm_parse_path)
    gm_parse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm_parse)
    read_gm_dict = gm_parse.read_gm_dict
    gm_parser2 = gm_parse.gm_parser2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import networkx as nx


class ABSDDataProcessor:
    """
    Process ABSD data to extract turnaround time features
    """
    
    def __init__(self, network_file, angles_file=None):
        self.network_file = network_file
        self.angles_file = angles_file
        self.nodes = {}
        self.edges = {}
        self.graph = None
        
    def load_network(self):
        """
        Load airport network from GM file
        """
        try:
            angles, nodes, edges, aircrafts = gm_parser2(self.network_file, self.angles_file)
            
            # Convert to dictionaries
            for node in nodes:
                if len(node) >= 7:
                    self.nodes[node[0]] = {
                        'id': node[0],
                        'x': float(node[1]) if node[1] else 0,
                        'y': float(node[2]) if node[2] else 0,
                        'lat': float(node[3]) if node[3] else 0,
                        'lon': float(node[4]) if node[4] else 0,
                        'specification': node[6] if len(node) > 6 else '',
                        'name': node[7] if len(node) > 7 else ''
                    }
            
            for edge in edges:
                if len(edge) >= 6:
                    self.edges[edge[0]] = {
                        'id': edge[0],
                        'start_node': edge[1],
                        'end_node': edge[2],
                        'length': float(edge[4]) if edge[4] else 0,
                        'specification': edge[5] if len(edge) > 5 else '',
                        'name': edge[7] if len(edge) > 7 else ''
                    }
            
            print(f"Loaded {len(self.nodes)} nodes and {len(self.edges)} edges")
            return aircrafts
            
        except Exception as e:
            print(f"Error loading network: {e}")
            return []
    
    def build_graph(self):
        """
        Build NetworkX graph from nodes and edges
        """
        self.graph = nx.Graph()
        
        for node_id, node_data in self.nodes.items():
            self.graph.add_node(node_id, **node_data)
        
        for edge_id, edge_data in self.edges.items():
            self.graph.add_edge(
                edge_data['start_node'],
                edge_data['end_node'],
                weight=edge_data['length'],
                **edge_data
            )
        
        print(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def calculate_path_distance(self, start_node, end_node):
        """
        Calculate shortest path distance between two nodes
        """
        if self.graph is None:
            self.build_graph()
        
        try:
            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
            distance = nx.shortest_path_length(self.graph, start_node, end_node, weight='weight')
            return distance, path
        except:
            return None, None
    
    def extract_gate_features(self):
        """
        Extract features about gates
        """
        gates = {}
        for node_id, node_data in self.nodes.items():
            if node_data['specification'] == 'gate':
                gates[node_id] = {
                    'gate_id': node_id,
                    'gate_name': node_data['name'],
                    'lat': node_data['lat'],
                    'lon': node_data['lon']
                }
        
        return pd.DataFrame(list(gates.values()))
    
    def process_aircraft_data(self, aircrafts):
        """
        Process aircraft movement data to extract turnaround features
        """
        aircraft_records = []
        
        for aircraft in aircrafts:
            if len(aircraft) < 6:
                continue
            
            try:
                record = {
                    'aircraft_id': aircraft[0] if len(aircraft) > 0 else None,
                    'aircraft_type': aircraft[1] if len(aircraft) > 1 else None,
                    'start_node': aircraft[2] if len(aircraft) > 2 else None,
                    'end_node': aircraft[3] if len(aircraft) > 3 else None,
                    'weight_class': aircraft[10] if len(aircraft) > 10 else None,
                }
                
                # Parse start and end times
                if len(aircraft) > 4 and isinstance(aircraft[4], list) and len(aircraft[4]) >= 3:
                    record['start_time'] = f"{aircraft[4][0]}:{aircraft[4][1]}:{aircraft[4][2]}"
                
                if len(aircraft) > 5 and isinstance(aircraft[5], list) and len(aircraft[5]) >= 3:
                    record['end_time'] = f"{aircraft[5][0]}:{aircraft[5][1]}:{aircraft[5][2]}"
                
                # Calculate turnaround time if both times available
                if 'start_time' in record and 'end_time' in record:
                    try:
                        start = datetime.strptime(record['start_time'], "%H:%M:%S")
                        end = datetime.strptime(record['end_time'], "%H:%M:%S")
                        duration = (end - start).total_seconds() / 60  # minutes
                        if duration < 0:
                            duration += 24 * 60  # Handle day rollover
                        record['turnaround_time'] = duration
                    except:
                        pass
                
                # Get node specifications
                if record['start_node'] in self.nodes:
                    record['start_node_type'] = self.nodes[record['start_node']]['specification']
                
                if record['end_node'] in self.nodes:
                    record['end_node_type'] = self.nodes[record['end_node']]['specification']
                
                # Calculate path distance
                if record['start_node'] and record['end_node']:
                    distance, path = self.calculate_path_distance(record['start_node'], record['end_node'])
                    if distance:
                        record['path_distance'] = distance
                        record['path_length'] = len(path)
                
                aircraft_records.append(record)
                
            except Exception as e:
                print(f"Error processing aircraft: {e}")
                continue
        
        return pd.DataFrame(aircraft_records)


def main():
    print("=" * 60)
    print("ABSD Data Processing for Turnaround Prediction")
    print("=" * 60)
    
    # Initialize processor
    network_file = 'Processed ABSD data/man_map (2).txt'
    processor = ABSDDataProcessor(network_file)
    
    # Load network
    print("\n1. Loading airport network...")
    aircrafts = processor.load_network()
    
    # Build graph
    print("\n2. Building network graph...")
    processor.build_graph()
    
    # Extract gate features
    print("\n3. Extracting gate features...")
    gates_df = processor.extract_gate_features()
    print(f"   Found {len(gates_df)} gates")
    if len(gates_df) > 0:
        print(gates_df.head())
    
    # Process aircraft data
    print("\n4. Processing aircraft movement data...")
    aircraft_df = processor.process_aircraft_data(aircrafts)
    print(f"   Processed {len(aircraft_df)} aircraft movements")
    
    if len(aircraft_df) > 0:
        print("\n   Sample aircraft data:")
        print(aircraft_df.head())
        
        # Save processed data
        aircraft_df.to_csv('processed_aircraft_data.csv', index=False)
        gates_df.to_csv('processed_gates_data.csv', index=False)
        print("\n5. Data saved:")
        print("   - processed_aircraft_data.csv")
        print("   - processed_gates_data.csv")
        
        # Statistics
        if 'turnaround_time' in aircraft_df.columns:
            valid_turnarounds = aircraft_df['turnaround_time'].dropna()
            if len(valid_turnarounds) > 0:
                print("\n6. Turnaround Time Statistics:")
                print(f"   - Count: {len(valid_turnarounds)}")
                print(f"   - Mean:  {valid_turnarounds.mean():.2f} minutes")
                print(f"   - Std:   {valid_turnarounds.std():.2f} minutes")
                print(f"   - Min:   {valid_turnarounds.min():.2f} minutes")
                print(f"   - Max:   {valid_turnarounds.max():.2f} minutes")
    else:
        print("\n   No aircraft data found in the network file.")
        print("   The model will use synthetic data for demonstration.")
    
    print("\n" + "=" * 60)
    print("Data processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
