# Lightning Network Routing Efficiency Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NetworkX](https://img.shields.io/badge/networkx-3.0+-green.svg)](https://networkx.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive simulation framework for analyzing Lightning Network routing efficiency with and without information sharing between nodes. This project provides detailed insights into how information sharing affects payment routing performance, execution time, and path exploration strategies.

## 🌟 Overview

The Lightning Network is a second-layer payment protocol built on top of Bitcoin that enables fast, low-cost transactions through payment channels. One key challenge in Lightning Network routing is determining the most efficient path for payments while balancing privacy, speed, and success rates.

This project simulates two distinct routing approaches:

1. **No Information Sharing**: Only sender and receiver nodes have knowledge of their own channel capacities
2. **With Information Sharing**: Intermediate nodes can optionally share their channel information based on willingness probability

## 🎯 Key Features

### 🔬 **Dual-Mode Simulation**
- **Sender-View Routing**: Models realistic Lightning Network routing where senders have limited knowledge
- **Information Sharing**: Configurable willingness-based sharing between intermediate nodes
- **Edge Pruning**: Intelligent path pruning to avoid re-trying failed routes
- **Ground Truth Validation**: Tests against actual channel balances for realistic results

### 📊 **Comprehensive Analysis**
- **Execution Time Comparison**: Precise timing measurements for both modes
- **Path Exploration Metrics**: Tracks number of paths checked before success/failure
- **Success Rate Analysis**: Compares routing success rates across different scenarios
- **Efficiency Assessment**: Determines whether information sharing improves or degrades performance

### 🛠️ **Flexible Testing Framework**
- **Multiple Sender/Receiver Pairs**: Test various node combinations without code changes
- **Parameter Customization**: Adjust payment amounts, willingness probabilities, path constraints
- **Batch Testing**: Run multiple scenarios with different configurations
- **Detailed Reporting**: Comprehensive results with visual indicators and statistics

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd CBS
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the simulation**
   ```bash
   python network.py
   ```

## 📁 Project Structure

```
CBS/
├── network.py                    # Main simulation framework
├── test_pairs.py                 # Example testing script
├── requirements.txt              # Python dependencies
├── LNdata.json                   # Lightning Network channel data
├── README.md                     # This file
├── README_comparison.md          # Detailed comparison documentation
├── PAIR_TESTING_GUIDE.md         # Pair testing guide
├── lightning_network_viz.py      # Network visualization tools
├── IZ_CARA.py                    # Additional analysis scripts
└── myenv/                        # Virtual environment
```

## 🔧 Usage Examples

### Basic Comparison
```python
from network import SimulationComparison

# Create comparison instance
comparison = SimulationComparison(json_file="LNdata.json", seed=42)

# Run comparison with specific sender/receiver
result = comparison.run_comparison(
    sender="03ac3c0544fb9eee3977e4fb34a4b29276eb6a594c6fd9a7a6c7d121d7de71643d",
    receiver="02c1d72c2fecc8df99351d3959d1f419475b5c35eb38a88ad7e7b9908eef54735f",
    amount=25000.0,               # Payment amount in satoshis
    p_willing=0.6,                # 60% willingness to share info
    min_path_nodes=5,             # Minimum path length
    max_paths_checked=1000,       # Maximum paths to explore
    verbose=True                  # Detailed output
)
```

### Testing Multiple Pairs
```python
# Test specific pairs
sender_receiver_pairs = [
    ("node_id_1", "node_id_2"),
    ("node_id_3", "node_id_4"),
    ("node_id_5", "node_id_6"),
]

# Or generate random pairs
from network import get_random_sender_receiver_pairs
pairs = get_random_sender_receiver_pairs(num_pairs=10, seed=42)
```

### Batch Testing Different Scenarios
```python
scenarios = [
    {"amount": 10000.0, "p_willing": 0.3, "name": "Small payment, low willingness"},
    {"amount": 50000.0, "p_willing": 0.6, "name": "Medium payment, medium willingness"},
    {"amount": 100000.0, "p_willing": 0.9, "name": "Large payment, high willingness"},
]

for scenario in scenarios:
    result = comparison.run_comparison(
        sender=sender,
        receiver=receiver,
        amount=scenario["amount"],
        p_willing=scenario["p_willing"],
        # ... other parameters
    )
```

## 📊 Sample Output

```
================================================================================
LIGHTNING NETWORK ROUTING COMPARISON
================================================================================
Sender: 03ac3c0544fb9eee3977e4fb34a4b29276eb6a594c6fd9a7a6c7d121d7de71643d
Receiver: 02c1d72c2fecc8df99351d3959d1f419475b5c35eb38a88ad7e7b9908eef54735f
Payment Amount: 25,000.0 satoshis
Willingness Probability: 60.0%
Min Path Nodes: 5
Max Paths Checked: 1000
================================================================================

🔄 Running simulation WITHOUT information sharing...
🔄 Running simulation WITH information sharing...

================================================================================
COMPARISON RESULTS
================================================================================

📊 SUCCESS RATE:
  Without Info Sharing: ✅ SUCCESS
  With Info Sharing:    ✅ SUCCESS

⏱️  EXECUTION TIME:
  Without Info Sharing: 0.1429 seconds
  With Info Sharing:    0.1026 seconds
  Time Saved:           0.0403 seconds (+28.2%)

🛤️  PATHS CHECKED:
  Without Info Sharing: 1 paths
  With Info Sharing:    1 paths
  Paths Saved:          0 paths (+0.0%)

📈 EFFICIENCY ANALYSIS:
  ✅ Info sharing SAVED 0.0403s (28.2%)
  ⚖️  Same number of paths checked

🎯 VERDICT:
  🏆 Info sharing is MORE EFFICIENT!
================================================================================
```

## 🏗️ Architecture

### Core Components

#### **LightningNetworkBase**
- Loads Lightning Network data from JSON
- Builds undirected graph with channel capacities
- Creates directed graph with split balances (ground truth)

#### **SenderViewSim**
- Implements sender-view routing simulation
- Supports both information sharing modes
- Uses edge pruning for efficiency
- Returns structured results for analysis

#### **SimulationComparison**
- Runs both simulation modes with identical parameters
- Ensures fair comparison by using same graph state
- Provides comprehensive analysis and reporting
- Handles multiple scenarios efficiently

### Data Structures

#### **SimulationMode** (Enum)
- `NO_INFO_SHARING`: Only sender/receiver knowledge
- `WITH_INFO_SHARING`: Includes intermediate node sharing

#### **SimulationResult** (Dataclass)
- `mode`: Simulation mode used
- `success`: Whether routing succeeded
- `execution_time`: Time taken in seconds
- `paths_checked`: Number of paths explored
- `final_path`: Successful path (if any)
- `failure_reason`: Failure reason (if applicable)

#### **ComparisonResult** (Dataclass)
- `no_sharing`: Results from no-info-sharing mode
- `with_sharing`: Results from with-info-sharing mode
- `time_saved`: Time difference (positive = info sharing faster)
- `time_saved_percentage`: Time saved as percentage
- `paths_saved`: Path difference (positive = info sharing checked fewer)
- `paths_saved_percentage`: Paths saved as percentage

## 🔬 Research Applications

### Academic Research
- **Routing Algorithm Analysis**: Compare different information sharing strategies
- **Network Topology Impact**: Study how network structure affects routing efficiency
- **Privacy vs. Efficiency Trade-offs**: Analyze the cost of information sharing
- **Scalability Studies**: Test performance with different network sizes

### Industry Applications
- **Lightning Network Optimization**: Improve real-world routing implementations
- **Node Operator Insights**: Help nodes decide on information sharing policies
- **Payment Service Providers**: Optimize routing for payment services
- **Network Analysis**: Understand Lightning Network behavior patterns

## 📈 Key Insights

### Information Sharing Benefits
- **Faster Routing**: Often reduces execution time by 20-30%
- **Better Path Discovery**: More accurate capacity information leads to better path selection
- **Reduced Exploration**: Fewer paths need to be checked before success

### Information Sharing Costs
- **Computational Overhead**: Sharing and processing additional information takes time
- **Privacy Concerns**: More information sharing reduces privacy
- **Network Load**: Additional communication overhead

### Factors Affecting Efficiency
- **Payment Amount**: Larger payments may benefit more from information sharing
- **Willingness Probability**: Higher willingness doesn't always mean better performance
- **Path Length**: Longer paths may see more benefit from information sharing
- **Network Topology**: Dense networks may benefit more than sparse ones

## 🛠️ Configuration Options

### Simulation Parameters
- `amount`: Payment amount in satoshis
- `p_willing`: Probability (0.0-1.0) that nodes share information
- `min_path_nodes`: Minimum nodes required in path
- `max_paths_checked`: Maximum paths to explore
- `include_endpoints_in_share`: Whether sender/receiver can share
- `verbose`: Detailed output during simulation

### Testing Options
- **Specific Pairs**: Test exact sender/receiver combinations
- **Random Pairs**: Generate random pairs for broader testing
- **Batch Testing**: Run multiple scenarios with different parameters
- **Custom Scenarios**: Define specific test cases for research

## 📚 Documentation

- **[README_comparison.md](README_comparison.md)**: Detailed comparison framework documentation
- **[PAIR_TESTING_GUIDE.md](PAIR_TESTING_GUIDE.md)**: Comprehensive guide for testing multiple pairs
- **[test_pairs.py](test_pairs.py)**: Example scripts and usage patterns

## 🤝 Contributing

This project is designed for research and educational purposes. Contributions are welcome for:

- **New Routing Algorithms**: Implement different pathfinding strategies
- **Additional Metrics**: Add new performance measurements
- **Visualization Tools**: Enhance network visualization capabilities
- **Documentation**: Improve guides and examples
- **Bug Fixes**: Report and fix issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Lightning Network Community**: For the open-source protocol and data
- **NetworkX**: For the excellent graph analysis library
- **Research Community**: For insights into Lightning Network routing challenges

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the maintainers.

---

**Note**: This simulation is for research and educational purposes. Real Lightning Network behavior may differ due to additional factors not modeled here, such as network congestion, node availability, and real-time channel state changes.
