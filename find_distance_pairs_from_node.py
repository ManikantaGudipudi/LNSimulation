#!/usr/bin/env python3
"""
Utility to find all node pairs with specific shortest distances from a given source node.
"""

import json
import networkx as nx
from typing import List, Tuple, Dict
from collections import defaultdict
from network import LightningNetworkBase

class DistancePairsFromNode:
    """Finds all nodes at specific distances from a given source node."""
    
    def __init__(self, json_file: str = "LNdata.json"):
        self.json_file = json_file
        self.base_sim = None
        
    def _setup_network(self) -> None:
        """Load and build the Lightning Network graph."""
        if self.base_sim is None:
            self.base_sim = LightningNetworkBase(json_file=self.json_file, seed=42)
            self.base_sim.load_data()
            self.base_sim.build_undirected()
            print(f"Network loaded: {self.base_sim.G.number_of_nodes()} nodes, {self.base_sim.G.number_of_edges()} edges")
    
    def find_all_distances_from_node(self, source_node: str) -> Dict[int, List[str]]:
        """
        Find all nodes at each distance from the source node.
        
        Args:
            source_node: The source node to find distances from
            
        Returns:
            Dictionary mapping distance -> list of nodes at that distance
        """
        self._setup_network()
        
        if source_node not in self.base_sim.G:
            raise ValueError(f"Source node {source_node} not found in the network")
        
        print(f"Finding all distances from node: {source_node}")
        
        # Use BFS to find all distances
        distances = {}
        visited = set()
        queue = [(source_node, 0)]
        
        while queue:
            current_node, distance = queue.pop(0)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            if distance not in distances:
                distances[distance] = []
            distances[distance].append(current_node)
            
            # Add neighbors to queue
            for neighbor in self.base_sim.G.neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return distances
    
    def find_pairs_at_distances(self, source_node: str, max_distance: int = None) -> Dict[int, List[Tuple[str, str]]]:
        """
        Find all node pairs with specific distances from source node.
        
        Args:
            source_node: The source node to find distances from
            max_distance: Maximum distance to search (None for all possible)
            
        Returns:
            Dictionary mapping distance -> list of (source, target) pairs
        """
        self._setup_network()
        
        if source_node not in self.base_sim.G:
            raise ValueError(f"Source node {source_node} not found in the network")
        
        print(f"Finding pairs at distances from node: {source_node}")
        
        # Get all distances
        distance_nodes = self.find_all_distances_from_node(source_node)
        
        # Find pairs for each distance
        pairs_by_distance = {}
        
        for distance in sorted(distance_nodes.keys()):
            if max_distance is not None and distance > max_distance:
                break
                
            if distance == 0:
                # Distance 0 is the source node itself
                pairs_by_distance[distance] = [(source_node, source_node)]
            else:
                # All nodes at this distance
                target_nodes = distance_nodes[distance]
                pairs = [(source_node, target) for target in target_nodes]
                pairs_by_distance[distance] = pairs
                
                print(f"Distance {distance}: {len(target_nodes)} nodes")
        
        return pairs_by_distance
    
    def get_distance_statistics(self, source_node: str) -> Dict:
        """
        Get statistics about distances from the source node.
        
        Args:
            source_node: The source node to analyze
            
        Returns:
            Dictionary with distance statistics
        """
        self._setup_network()
        
        if source_node not in self.base_sim.G:
            raise ValueError(f"Source node {source_node} not found in the network")
        
        distance_nodes = self.find_all_distances_from_node(source_node)
        
        stats = {
            'source_node': source_node,
            'max_distance': max(distance_nodes.keys()) if distance_nodes else 0,
            'total_reachable_nodes': sum(len(nodes) for nodes in distance_nodes.values()),
            'distance_counts': {d: len(nodes) for d, nodes in distance_nodes.items()},
            'total_nodes_in_network': self.base_sim.G.number_of_nodes()
        }
        
        return stats
    
    def save_pairs_to_file(self, pairs_by_distance: Dict[int, List[Tuple[str, str]]], 
                          filename: str = "distance_pairs.txt") -> None:
        """
        Save pairs to a file.
        
        Args:
            pairs_by_distance: Dictionary mapping distance -> list of pairs
            filename: Output filename
        """
        with open(filename, "w") as f:
            f.write("# Distance pairs from source node\n")
            f.write("# Format: distance,source,target\n")
            
            for distance in sorted(pairs_by_distance.keys()):
                for source, target in pairs_by_distance[distance]:
                    f.write(f"{distance},{source},{target}\n")
        
        total_pairs = sum(len(pairs) for pairs in pairs_by_distance.values())
        print(f"Saved {total_pairs} pairs to '{filename}'")

def main():
    """Example usage with the specified node."""
    finder = DistancePairsFromNode(json_file="LNdata.json")
    
    # The specified source node
    source_node = "02dd04e277bf2ee7e8cd8e9766d5615005ac19fa872be8cdfe612a9950ada27b70"
    
    print("=" * 80)
    print("DISTANCE PAIRS FROM SPECIFIC NODE")
    print("=" * 80)
    print(f"Source node: {source_node}")
    print("=" * 80)
    
    try:
        # Get statistics first
        print("\n1. Getting distance statistics...")
        stats = finder.get_distance_statistics(source_node)
        
        print(f"\nDistance Statistics:")
        print(f"  Source node: {stats['source_node']}")
        print(f"  Max distance: {stats['max_distance']}")
        print(f"  Total reachable nodes: {stats['total_reachable_nodes']}")
        print(f"  Total nodes in network: {stats['total_nodes_in_network']}")
        print(f"  Reachability: {(stats['total_reachable_nodes']/stats['total_nodes_in_network']*100):.1f}%")
        
        print(f"\nDistance Distribution:")
        for distance in sorted(stats['distance_counts'].keys()):
            count = stats['distance_counts'][distance]
            print(f"  Distance {distance}: {count} nodes")
        
        # Find pairs at all distances
        print(f"\n2. Finding pairs at all distances...")
        pairs_by_distance = finder.find_pairs_at_distances(source_node)
        
        print(f"\nFound pairs at distances:")
        total_pairs = 0
        for distance in sorted(pairs_by_distance.keys()):
            pairs = pairs_by_distance[distance]
            print(f"  Distance {distance}: {len(pairs)} pairs")
            total_pairs += len(pairs)
            
            # Show first few examples for each distance
            if len(pairs) <= 5:
                for source, target in pairs:
                    print(f"    {source[:20]}... -> {target[:20]}...")
            else:
                print(f"    Examples:")
                for source, target in pairs[:3]:
                    print(f"      {source[:20]}... -> {target[:20]}...")
                print(f"      ... and {len(pairs)-3} more")
        
        print(f"\nTotal pairs found: {total_pairs}")
        
        # Save to file
        print(f"\n3. Saving pairs to file...")
        finder.save_pairs_to_file(pairs_by_distance, f"distance_pairs_from_{source_node[:20]}.txt")
        
        # Show some specific distance examples
        print(f"\n4. Specific distance examples:")
        for distance in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if distance in pairs_by_distance:
                pairs = pairs_by_distance[distance]
                print(f"\nDistance {distance} pairs (showing first 5):")
                for i, (source, target) in enumerate(pairs[:5]):
                    print(f"  {i+1}. {source} -> {target}")
                if len(pairs) > 5:
                    print(f"  ... and {len(pairs)-5} more")
            else:
                print(f"\nNo nodes at distance {distance}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()


