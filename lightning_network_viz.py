#!/usr/bin/env python3
"""
Lightning Network Visualization Tool

This script loads Lightning Network channel data and creates an interactive
network visualization showing nodes (peers) and channels with their capacities.
"""

import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import argparse
import sys
from datetime import datetime
import random

class LightningNetworkAnalyzer:
    """Analyzes and visualizes Lightning Network data"""
    
    def __init__(self, json_file="LNdata.json"):
        self.json_file = json_file
        self.G = nx.Graph()
        self.G_dir = nx.DiGraph()
        self.channels_data = []
        self.stats = {}
        self.dir_stats = {}
        
    def load_data(self):
        """Load channel data from JSON file"""
        print("Loading Lightning Network data...")
        try:
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            self.channels_data = data.get('channels', [])
            print(f"✓ Loaded {len(self.channels_data)} channels")
            return True
            
        except FileNotFoundError:
            print(f"Error: Could not find file '{self.json_file}'")
            return False
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {e}")
            return False
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def build_network(self, min_capacity=1000000, active_only=True):
        """
        Build network graph from channel data
        
        Args:
            min_capacity: Minimum channel capacity in satoshis to include
            active_only: Only include active channels
        """
        print("Building network graph...")
        
        # Clear existing graph
        self.G.clear()
        
        # Filter channels based on criteria
        filtered_channels = []
        for channel in self.channels_data:
            if active_only and not channel.get('active', False):
                continue
            if channel.get('satoshis', 0) < min_capacity:
                continue
            filtered_channels.append(channel)
        
        print(f"Using {len(filtered_channels)} channels (min capacity: {min_capacity:,} sats)")
        
        # Add nodes and edges
        for channel in filtered_channels:
            source = channel.get('source', '')
            destination = channel.get('destination', '')
            
            if not source or not destination:
                continue
                
            # Add nodes with basic info
            if not self.G.has_node(source):
                self.G.add_node(source, node_type='peer')
            if not self.G.has_node(destination):
                self.G.add_node(destination, node_type='peer')
            
            # Add edge with channel information
            edge_data = {
                'capacity': channel.get('satoshis', 0),
                'capacity_msat': channel.get('amount_msat', '0msat'),
                'base_fee': channel.get('base_fee_millisatoshi', 0),
                'fee_rate': channel.get('fee_per_millionth', 0),
                'delay': channel.get('delay', 0),
                'active': channel.get('active', False),
                'public': channel.get('public', False),
                'short_channel_id': channel.get('short_channel_id', ''),
                'last_update': channel.get('last_update', 0)
            }
            
            self.G.add_edge(source, destination, **edge_data)
        
        print(f"✓ Network built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Calculate network statistics
        self._calculate_stats()
        
        # Create directional graph
        self._create_directional_graph()
        
    def _calculate_stats(self):
        """Calculate network statistics"""
        if self.G.number_of_nodes() == 0:
            return
            
        # Basic stats
        self.stats = {
            'total_nodes': self.G.number_of_nodes(),
            'total_edges': self.G.number_of_edges(),
            'avg_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'density': nx.density(self.G),
            'is_connected': nx.is_connected(self.G),
            'num_components': nx.number_connected_components(self.G)
        }
        
        # Capacity stats
        capacities = [data['capacity'] for _, _, data in self.G.edges(data=True)]
        if capacities:
            self.stats.update({
                'total_capacity': sum(capacities),
                'avg_capacity': np.mean(capacities),
                'median_capacity': np.median(capacities),
                'max_capacity': max(capacities),
                'min_capacity': min(capacities)
            })
        
        # Fee stats
        base_fees = [data['base_fee'] for _, _, data in self.G.edges(data=True)]
        fee_rates = [data['fee_rate'] for _, _, data in self.G.edges(data=True)]
        
        if base_fees:
            self.stats.update({
                'avg_base_fee': np.mean(base_fees),
                'avg_fee_rate': np.mean(fee_rates)
            })
    
    def _create_directional_graph(self):
        """Create directional graph by randomly splitting channel capacities"""
        print("Creating directional graph with random capacity splitting...")
        
        # Clear existing directional graph
        self.G_dir.clear()
        
        # Copy all nodes from undirected graph
        for node in self.G.nodes():
            self.G_dir.add_node(node, **self.G.nodes[node])
        
        forward_count = 0
        backward_count = 0
        
        # For each edge in undirected graph, create two directed edges
        for u, v, data in self.G.edges(data=True):
            total_capacity = data['capacity']
            
            # Randomly split capacity between the two directions
            # Ensure both directions have at least 1 satoshi capacity
            capacity_forward = random.randint(1, total_capacity - 1)
            capacity_backward = total_capacity - capacity_forward
            
            # Create forward edge (u -> v)
            forward_data = data.copy()
            forward_data['capacity'] = capacity_forward
            forward_data['direction'] = 'forward'
            forward_data['original_total_capacity'] = total_capacity
            self.G_dir.add_edge(u, v, **forward_data)
            forward_count += 1
            
            # Create backward edge (v -> u)
            backward_data = data.copy()
            backward_data['capacity'] = capacity_backward
            backward_data['direction'] = 'backward'
            backward_data['original_total_capacity'] = total_capacity
            self.G_dir.add_edge(v, u, **backward_data)
            backward_count += 1
        
        print(f"✓ Directional graph created: {self.G_dir.number_of_nodes()} nodes, {self.G_dir.number_of_edges()} directed edges")
        print(f"  - {forward_count} forward edges, {backward_count} backward edges")
        
        # Calculate directional graph statistics
        self._calculate_dir_stats()
    
    def _calculate_dir_stats(self):
        """Calculate directional graph statistics"""
        if self.G_dir.number_of_nodes() == 0:
            return
            
        # Basic stats for directional graph
        self.dir_stats = {
            'total_nodes': self.G_dir.number_of_nodes(),
            'total_edges': self.G_dir.number_of_edges(),
            'avg_in_degree': sum(dict(self.G_dir.in_degree()).values()) / self.G_dir.number_of_nodes(),
            'avg_out_degree': sum(dict(self.G_dir.out_degree()).values()) / self.G_dir.number_of_nodes(),
            'is_weakly_connected': nx.is_weakly_connected(self.G_dir),
            'is_strongly_connected': nx.is_strongly_connected(self.G_dir),
            'num_weak_components': nx.number_weakly_connected_components(self.G_dir),
            'num_strong_components': nx.number_strongly_connected_components(self.G_dir)
        }
        
        # Capacity stats for directional graph
        capacities = [data['capacity'] for _, _, data in self.G_dir.edges(data=True)]
        if capacities:
            self.dir_stats.update({
                'total_capacity': sum(capacities),
                'avg_capacity': np.mean(capacities),
                'median_capacity': np.median(capacities),
                'max_capacity': max(capacities),
                'min_capacity': min(capacities)
            })
        
        # Fee stats for directional graph
        base_fees = [data['base_fee'] for _, _, data in self.G_dir.edges(data=True)]
        fee_rates = [data['fee_rate'] for _, _, data in self.G_dir.edges(data=True)]
        
        if base_fees:
            self.dir_stats.update({
                'avg_base_fee': np.mean(base_fees),
                'avg_fee_rate': np.mean(fee_rates)
            })
    
    def print_stats(self):
        """Print network statistics"""
        print("\n" + "="*50)
        print("LIGHTNING NETWORK STATISTICS")
        print("="*50)
        
        print(f"Nodes (Peers): {self.stats.get('total_nodes', 0):,}")
        print(f"Channels: {self.stats.get('total_edges', 0):,}")
        print(f"Average degree: {self.stats.get('avg_degree', 0):.2f}")
        print(f"Network density: {self.stats.get('density', 0):.6f}")
        print(f"Connected: {self.stats.get('is_connected', False)}")
        print(f"Components: {self.stats.get('num_components', 0)}")
        
        if 'total_capacity' in self.stats:
            print(f"\nCapacity Statistics:")
            print(f"Total capacity: {self.stats['total_capacity']:,} sats ({self.stats['total_capacity']/1e8:.2f} BTC)")
            print(f"Average channel capacity: {self.stats['avg_capacity']:,.0f} sats")
            print(f"Median channel capacity: {self.stats['median_capacity']:,.0f} sats")
            print(f"Largest channel: {self.stats['max_capacity']:,} sats")
            print(f"Smallest channel: {self.stats['min_capacity']:,} sats")
        
        if 'avg_base_fee' in self.stats:
            print(f"\nFee Statistics:")
            print(f"Average base fee: {self.stats['avg_base_fee']:,.0f} msats")
            print(f"Average fee rate: {self.stats['avg_fee_rate']:,.0f} ppm")
        
        # Print directional graph statistics
        if self.dir_stats:
            print(f"\n" + "="*50)
            print("DIRECTIONAL GRAPH STATISTICS")
            print("="*50)
            
            print(f"Nodes: {self.dir_stats.get('total_nodes', 0):,}")
            print(f"Directed Edges: {self.dir_stats.get('total_edges', 0):,}")
            print(f"Average in-degree: {self.dir_stats.get('avg_in_degree', 0):.2f}")
            print(f"Average out-degree: {self.dir_stats.get('avg_out_degree', 0):.2f}")
            print(f"Weakly connected: {self.dir_stats.get('is_weakly_connected', False)}")
            print(f"Strongly connected: {self.dir_stats.get('is_strongly_connected', False)}")
            print(f"Weak components: {self.dir_stats.get('num_weak_components', 0)}")
            print(f"Strong components: {self.dir_stats.get('num_strong_components', 0)}")
            
            if 'total_capacity' in self.dir_stats:
                print(f"\nDirectional Capacity Statistics:")
                print(f"Total capacity: {self.dir_stats['total_capacity']:,} sats ({self.dir_stats['total_capacity']/1e8:.2f} BTC)")
                print(f"Average edge capacity: {self.dir_stats['avg_capacity']:,.0f} sats")
                print(f"Median edge capacity: {self.dir_stats['median_capacity']:,.0f} sats")
                print(f"Largest edge: {self.dir_stats['max_capacity']:,} sats")
                print(f"Smallest edge: {self.dir_stats['min_capacity']:,} sats")
    
    def create_network_visualization(self, max_nodes=1000, layout='spring'):
        """
        Create interactive network visualization
        
        Args:
            max_nodes: Maximum number of nodes to display (for performance)
            layout: Layout algorithm ('spring', 'circular', 'random')
        """
        print(f"Creating network visualization (max {max_nodes} nodes)...")
        
        # Sample nodes if too many
        if self.G.number_of_nodes() > max_nodes:
            # Get top nodes by degree
            degrees = dict(self.G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node for node, _ in top_nodes]
            
            # Create subgraph
            subgraph = self.G.subgraph(top_node_ids)
        else:
            subgraph = self.G
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.random_layout(subgraph)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge data
            edge_data = subgraph[edge[0]][edge[1]]
            capacity = edge_data.get('capacity', 0)
            fee_rate = edge_data.get('fee_rate', 0)
            
            edge_info.append(f"Channel: {edge[0][:8]}... ↔ {edge[1][:8]}...<br>"
                           f"Capacity: {capacity:,} sats<br>"
                           f"Fee Rate: {fee_rate:,} ppm")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node degree (number of connections)
            degree = subgraph.degree(node)
            node_sizes.append(max(5, min(20, degree * 2)))
            
            # Node info
            node_text.append(f"Node: {node[:12]}...")
            node_info.append(f"Node: {node}<br>"
                           f"Connections: {degree}<br>"
                           f"Type: Peer")
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            customdata=node_info,
            hovertemplate='%{customdata}<extra></extra>',
            marker=dict(
                size=node_sizes,
                color=node_sizes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Node Degree"),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text=f'Lightning Network Visualization<br><sub>Nodes: {subgraph.number_of_nodes()}, '
                                    f'Channels: {subgraph.number_of_edges()}</sub>',
                               font=dict(size=16)
                           ),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Node size represents number of connections",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def create_directional_visualization(self, max_nodes=500, layout='spring'):
        """
        Create interactive directional network visualization
        
        Args:
            max_nodes: Maximum number of nodes to display (for performance)
            layout: Layout algorithm ('spring', 'circular', 'random')
        """
        print(f"Creating directional network visualization (max {max_nodes} nodes)...")
        
        # Sample nodes if too many
        if self.G_dir.number_of_nodes() > max_nodes:
            print("Sampling top nodes by degree...")
            # Get top nodes by degree (in + out)
            in_degrees = dict(self.G_dir.in_degree())
            out_degrees = dict(self.G_dir.out_degree())
            total_degrees = {node: in_degrees[node] + out_degrees[node] 
                           for node in self.G_dir.nodes()}
            top_nodes = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node for node, _ in top_nodes]
            
            # Create subgraph that preserves directional pairs
            # Include all edges where at least one node is in the selected set
            subgraph = nx.DiGraph()
            for node in top_node_ids:
                if node in self.G_dir:
                    subgraph.add_node(node, **self.G_dir.nodes[node])
            
            # Add edges where at least one endpoint is in our selected nodes
            for u, v, data in self.G_dir.edges(data=True):
                if u in top_node_ids or v in top_node_ids:
                    subgraph.add_edge(u, v, **data)
        else:
            subgraph = self.G_dir
        
        # Calculate layout
        print("Calculating network layout...")
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=1, iterations=30)  # Reduced iterations
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.random_layout(subgraph)
        
        print("Preparing edge traces...")
        # Prepare edge traces with arrows
        edge_x = []
        edge_y = []
        edge_info = []
        edge_capacities = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Add edge line
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Get edge data
            edge_data = subgraph[edge[0]][edge[1]]
            capacity = edge_data.get('capacity', 0)
            fee_rate = edge_data.get('fee_rate', 0)
            direction = edge_data.get('direction', 'unknown')
            original_capacity = edge_data.get('original_total_capacity', capacity)
            
            edge_info.append(f"Channel: {edge[0][:8]}... → {edge[1][:8]}...<br>"
                           f"Direction: {direction}<br>"
                           f"Capacity: {capacity:,} sats<br>"
                           f"Original Total: {original_capacity:,} sats<br>"
                           f"Fee Rate: {fee_rate:,} ppm")
            
            edge_capacities.append(capacity)
        
        # Create edge traces with different colors and widths for direction
        # Split edges into forward and backward for different styling
        forward_edges = []
        backward_edges = []
        
        print(f"Processing {subgraph.number_of_edges()} edges for visualization...")
        
        for edge in subgraph.edges():
            edge_data = subgraph[edge[0]][edge[1]]
            direction = edge_data.get('direction', 'unknown')
            capacity = edge_data.get('capacity', 0)
            
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_info = {
                'x': [x0, x1, None],
                'y': [y0, y1, None],
                'capacity': capacity,
                'direction': direction,
                'info': f"Channel: {edge[0][:8]}... → {edge[1][:8]}...<br>"
                       f"Direction: {direction}<br>"
                       f"Capacity: {capacity:,} sats<br>"
                       f"Fee Rate: {edge_data.get('fee_rate', 0):,} ppm"
            }
            
            if direction == 'forward':
                forward_edges.append(edge_info)
            elif direction == 'backward':
                backward_edges.append(edge_info)
            else:
                print(f"Warning: Unknown direction '{direction}' for edge {edge}")
        
        print(f"Found {len(forward_edges)} forward edges and {len(backward_edges)} backward edges")
        
        # Create forward edge trace (green, thicker)
        forward_x = []
        forward_y = []
        forward_info = []
        
        for edge_info in forward_edges:
            forward_x.extend(edge_info['x'])
            forward_y.extend(edge_info['y'])
            forward_info.append(edge_info['info'])
        
        # Use a fixed width for forward edges (thicker)
        forward_trace = go.Scatter(
            x=forward_x, y=forward_y,
            line=dict(width=3, color='#2E8B57'),  # Green for forward, fixed width
            hoverinfo='none',
            mode='lines',
            name='Forward Direction',
            showlegend=True
        )
        
        # Create backward edge trace (red, thinner)
        backward_x = []
        backward_y = []
        backward_info = []
        
        for edge_info in backward_edges:
            backward_x.extend(edge_info['x'])
            backward_y.extend(edge_info['y'])
            backward_info.append(edge_info['info'])
        
        # Use a fixed width for backward edges (thinner)
        backward_trace = go.Scatter(
            x=backward_x, y=backward_y,
            line=dict(width=2, color='#DC143C'),  # Red for backward, fixed width
            hoverinfo='none',
            mode='lines',
            name='Backward Direction',
            showlegend=True
        )
        
        # Add arrow annotations for high-capacity edges only
        print("Adding directional arrows for high-capacity channels...")
        arrow_annotations = []
        max_arrows = min(20, subgraph.number_of_edges())  # Even fewer arrows for clarity
        
        # Sample high-capacity edges for arrows
        edge_capacities = [(edge, subgraph[edge[0]][edge[1]].get('capacity', 0)) 
                          for edge in subgraph.edges()]
        edge_capacities.sort(key=lambda x: x[1], reverse=True)
        sampled_edges = [edge for edge, _ in edge_capacities[:max_arrows]]
        
        for edge in sampled_edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Calculate arrow position (slightly offset from midpoint)
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            
            # Calculate arrow direction
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normalize direction
                dx_norm = dx / length
                dy_norm = dy / length
                
                # Arrow size based on capacity
                capacity = subgraph[edge[0]][edge[1]].get('capacity', 0)
                arrow_size = max(0.03, min(0.08, np.log10(capacity + 1) * 0.01))
                
                arrow_annotations.append(dict(
                    x=mid_x,
                    y=mid_y,
                    ax=dx_norm * arrow_size,
                    ay=dy_norm * arrow_size,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=2,
                    arrowwidth=2,
                    arrowcolor='#000000'
                ))
        
        print("Preparing node traces...")
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node degree (in + out)
            in_degree = subgraph.in_degree(node)
            out_degree = subgraph.out_degree(node)
            total_degree = in_degree + out_degree
            node_sizes.append(max(5, min(20, total_degree * 2)))
            
            # Node info
            node_text.append(f"Node: {node[:12]}...")
            node_info.append(f"Node: {node}<br>"
                           f"In-degree: {in_degree}<br>"
                           f"Out-degree: {out_degree}<br>"
                           f"Total connections: {total_degree}")
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            customdata=node_info,
            hovertemplate='%{customdata}<extra></extra>',
            marker=dict(
                size=node_sizes,
                color=node_sizes,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Node Degree"),
                line=dict(width=2, color='white')
            )
        )
        
        print("Creating interactive figure...")
        # Create figure with both edge traces and node trace
        fig = go.Figure(data=[backward_trace, forward_trace, node_trace],
                       layout=go.Layout(
                           title=dict(
                               text=f'Lightning Network Directional Visualization<br><sub>Nodes: {subgraph.number_of_nodes()}, '
                                    f'Directed Edges: {subgraph.number_of_edges()}</sub>',
                               font=dict(size=16)
                           ),
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=arrow_annotations + [dict(
                               text="Green=Forward, Red=Backward, Arrows=High-capacity channels",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        return fig
    
    def create_capacity_analysis(self):
        """Create capacity distribution analysis"""
        print("Creating capacity analysis...")
        
        # Get capacity data
        capacities = [data['capacity'] for _, _, data in self.G.edges(data=True)]
        
        if not capacities:
            print("No capacity data available")
            return None
        
        # Create histogram
        fig = go.Figure()
        
        # Log scale for better visualization
        log_capacities = np.log10(capacities)
        
        fig.add_trace(go.Histogram(
            x=log_capacities,
            nbinsx=50,
            name='Channel Capacity Distribution',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Lightning Network Channel Capacity Distribution (Log Scale)',
            xaxis_title='Log10(Capacity in Satoshis)',
            yaxis_title='Number of Channels',
            showlegend=False,
            plot_bgcolor='white'
        )
        
        # Add statistics text
        stats_text = f"Total Channels: {len(capacities):,}<br>"
        stats_text += f"Total Capacity: {sum(capacities):,} sats ({sum(capacities)/1e8:.2f} BTC)<br>"
        stats_text += f"Average: {np.mean(capacities):,.0f} sats<br>"
        stats_text += f"Median: {np.median(capacities):,.0f} sats"
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def save_network_data(self, filename="lightning_network_analysis.json"):
        """Save network data and statistics to JSON file"""
        print(f"Saving network analysis to {filename}...")
        
        # Prepare data for JSON serialization
        network_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'nodes': list(self.G.nodes()),
            'edges': [
                {
                    'source': edge[0],
                    'destination': edge[1],
                    **edge[2]
                }
                for edge in self.G.edges(data=True)
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(network_data, f, indent=2)
        
        print(f"✓ Network data saved to {filename}")
    
    def save_directional_data(self, filename="lightning_directional_analysis.json"):
        """Save directional network data and statistics to JSON file"""
        print(f"Saving directional network analysis to {filename}...")
        
        # Prepare data for JSON serialization
        directional_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.dir_stats,
            'nodes': list(self.G_dir.nodes()),
            'edges': [
                {
                    'source': edge[0],
                    'destination': edge[1],
                    **edge[2]
                }
                for edge in self.G_dir.edges(data=True)
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(directional_data, f, indent=2)
        
        print(f"✓ Directional network data saved to {filename}")
    
    def simulate_paths(self, num_simulations=10):
        """
        Simulate random path finding between nodes in the bidirectional graph
        
        Args:
            num_simulations: Number of random path simulations to run
        """
        print(f"\n{'='*60}")
        print("LIGHTNING NETWORK PATH SIMULATION")
        print(f"{'='*60}")
        
        if self.G.number_of_nodes() < 2:
            print("Error: Need at least 2 nodes for path simulation")
            return
        
        # Get all nodes
        nodes = list(self.G.nodes())
        
        successful_paths = 0
        total_hops = 0
        path_lengths = []
        
        print(f"Running {num_simulations} random path simulations...")
        print(f"Network has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
        print(f"Network is connected: {nx.is_connected(self.G)}")
        print()
        
        for i in range(num_simulations):
            # Randomly select two different nodes
            source, target = random.sample(nodes, 2)
            
            try:
                # Find shortest path using NetworkX
                if nx.has_path(self.G, source, target):
                    path = nx.shortest_path(self.G, source, target)
                    path_length = len(path) - 1  # Number of hops (edges)
                    intermediate_nodes = len(path) - 2  # Nodes between source and target
                    
                    print(f"Simulation {i+1:2d}: {source}... → {target}...")
                    print(f"           Path length: {path_length} hops")
                    print(f"           Intermediate nodes: {intermediate_nodes}")
                    print(f"           Full path: {' → '.join([node[:8] + '...' for node in path])}")
                    print()
                    
                    successful_paths += 1
                    total_hops += path_length
                    path_lengths.append(path_length)
                else:
                    print(f"Simulation {i+1:2d}: {source} → {target}...")
                    print(f"           No path exists")
                    print()
                    
            except nx.NetworkXNoPath:
                print(f"Simulation {i+1:2d}: {source}... → {target}...")
                print(f"           No path exists")
                print()
            except Exception as e:
                print(f"Simulation {i+1:2d}: Error - {e}")
                print()
        
        # Calculate statistics
        if successful_paths > 0:
            avg_path_length = total_hops / successful_paths
            min_path_length = min(path_lengths)
            max_path_length = max(path_lengths)
            
            print(f"{'='*60}")
            print("PATH SIMULATION RESULTS")
            print(f"{'='*60}")
            print(f"Successful paths: {successful_paths}/{num_simulations} ({successful_paths/num_simulations*100:.1f}%)")
            print(f"Average path length: {avg_path_length:.2f} hops")
            print(f"Shortest path: {min_path_length} hops")
            print(f"Longest path: {max_path_length} hops")
            
            # Path length distribution
            path_length_counts = Counter(path_lengths)
            print(f"\nPath length distribution:")
            for length in sorted(path_length_counts.keys()):
                count = path_length_counts[length]
                percentage = count / successful_paths * 100
                print(f"  {length} hops: {count} paths ({percentage:.1f}%)")
        else:
            print(f"{'='*60}")
            print("PATH SIMULATION RESULTS")
            print(f"{'='*60}")
            print("No successful paths found in any simulation!")
            print("This might indicate the network is not well connected.")
    
    def find_specific_path(self, source, target):
        """
        Find path between specific nodes
        
        Args:
            source: Source node ID
            target: Target node ID
        """
        print(f"\n{'='*60}")
        print("SPECIFIC PATH FINDING")
        print(f"{'='*60}")
        
        if source not in self.G:
            print(f"Error: Source node '{source}' not found in network")
            return
        
        if target not in self.G:
            print(f"Error: Target node '{target}' not found in network")
            return
        
        if source == target:
            print(f"Source and target are the same node: {source}")
            return
        
        try:
            if nx.has_path(self.G, source, target):
                path = nx.shortest_path(self.G, source, target)
                path_length = len(path) - 1
                intermediate_nodes = len(path) - 2
                
                print(f"Path found from {source} to {target}")
                print(f"Path length: {path_length} hops")
                print(f"Intermediate nodes: {intermediate_nodes}")
                print(f"Full path: {' → '.join(path)}")
                
                # Show edge details for each hop
                print(f"\nEdge details:")
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.G[u][v]
                    capacity = edge_data.get('capacity', 0)
                    fee_rate = edge_data.get('fee_rate', 0)
                    print(f"  {u[:12]}... → {v[:12]}... (Capacity: {capacity:,} sats, Fee: {fee_rate:,} ppm)")
            else:
                print(f"No path exists from {source} to {target}")
                
        except Exception as e:
            print(f"Error finding path: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Lightning Network Visualization Tool')
    parser.add_argument('--file', default='LNdata.json', help='JSON file with channel data')
    parser.add_argument('--min-capacity', type=int, default=1000000, 
                       help='Minimum channel capacity to include (satoshis)')
    parser.add_argument('--max-nodes', type=int, default=500,
                       help='Maximum number of nodes to display')
    parser.add_argument('--active-only', action='store_true', default=True,
                       help='Only include active channels')
    parser.add_argument('--layout', choices=['spring', 'circular', 'random'], 
                       default='spring', help='Network layout algorithm')
    parser.add_argument('--save-data', action='store_true',
                       help='Save network data to JSON file')
    parser.add_argument('--create-directional', action='store_true',
                       help='Create and visualize directional graph')
    parser.add_argument('--simulate-paths', type=int, metavar='N',
                       help='Run N random path simulations between nodes')
    parser.add_argument('--find-path', nargs=2, metavar=('SOURCE', 'TARGET'),
                       help='Find path between specific nodes')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = LightningNetworkAnalyzer(args.file)
    
    # Load data
    if not analyzer.load_data():
        sys.exit(1)
    
    # Build network
    analyzer.build_network(
        min_capacity=args.min_capacity,
        active_only=args.active_only
    )
    
    # Print statistics
    analyzer.print_stats()
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Network visualization
    network_fig = analyzer.create_network_visualization(
        max_nodes=args.max_nodes,
        layout=args.layout
    )
    network_html = "lightning_network.html"
    network_fig.write_html(network_html)
    print(f"✓ Network visualization saved to {network_html}")
    
    # Capacity analysis
    capacity_fig = analyzer.create_capacity_analysis()
    if capacity_fig:
        capacity_html = "lightning_capacity_analysis.html"
        capacity_fig.write_html(capacity_html)
        print(f"✓ Capacity analysis saved to {capacity_html}")
    
    # Create directional visualization if requested
    if args.create_directional:
        print("\nCreating directional graph visualization...")
        dir_fig = analyzer.create_directional_visualization(
            max_nodes=args.max_nodes,
            layout=args.layout
        )
        dir_html = "lightning_directional_network.html"
        dir_fig.write_html(dir_html)
        print(f"✓ Directional network visualization saved to {dir_html}")
    
    # Run path simulations if requested
    if args.simulate_paths:
        analyzer.simulate_paths(args.simulate_paths)
    
    # Find specific path if requested
    if args.find_path:
        source, target = args.find_path
        analyzer.find_specific_path(source, target)
    
    # Save data if requested
    if args.save_data:
        analyzer.save_network_data()
        if args.create_directional:
            analyzer.save_directional_data()
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
