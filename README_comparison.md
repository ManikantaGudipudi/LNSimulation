# Lightning Network Routing Comparison

This enhanced version of the Lightning Network simulation allows you to compare routing performance with and without information sharing between nodes.

## 🚀 Key Features

### Two Simulation Modes
1. **No Info Sharing**: Only sender and receiver have knowledge of their own channels
2. **With Info Sharing**: Intermediate nodes can share their channel information based on willingness probability

### Comprehensive Comparison
- **Execution Time**: Measures how long each simulation takes
- **Paths Checked**: Counts how many paths were explored before success/failure
- **Success Rate**: Whether each mode successfully found a route
- **Efficiency Analysis**: Determines if info sharing saves time and reduces path exploration

## 📊 Usage

### Basic Comparison
```python
from network import SimulationComparison

# Create comparison instance
comparison = SimulationComparison(json_file="LNdata.json", seed=42)

# Run comparison
result = comparison.run_comparison(
    amount=70000.0,               # Payment amount in satoshis
    p_willing=0.6,                # Probability of willingness to share (60%)
    min_path_nodes=2,             # Minimum nodes per path
    max_paths_checked=1000,       # Maximum paths to check
    verbose=False                 # Set to True for detailed output
)
```

### Advanced Usage
```python
# Run with specific sender/receiver pair
result = comparison.run_comparison(
    amount=50000.0,
    p_willing=0.8,
    min_path_nodes=3,
    max_paths_checked=500,
    seed_sender_receiver=("node1", "node2"),  # Fixed pair
    verbose=True
)
```

## 🏗️ Architecture

### Core Classes

#### `SimulationMode` (Enum)
- `NO_INFO_SHARING`: Only sender/receiver knowledge
- `WITH_INFO_SHARING`: Includes intermediate node sharing

#### `SimulationResult` (Dataclass)
- `mode`: Which simulation mode was used
- `success`: Whether routing succeeded
- `execution_time`: Time taken in seconds
- `paths_checked`: Number of paths explored
- `final_path`: The successful path (if any)
- `failure_reason`: Why it failed (if applicable)

#### `ComparisonResult` (Dataclass)
- `no_sharing`: Results from no-info-sharing simulation
- `with_sharing`: Results from with-info-sharing simulation
- `time_saved`: Time difference (positive = info sharing faster)
- `time_saved_percentage`: Time saved as percentage
- `paths_saved`: Path difference (positive = info sharing checked fewer)
- `paths_saved_percentage`: Paths saved as percentage

#### `SenderViewSim` (Enhanced)
- Configurable simulation mode via constructor
- Returns structured results instead of just printing
- Supports verbose/quiet modes
- Maintains backward compatibility

#### `SimulationComparison` (New)
- Runs both simulation modes with identical parameters
- Ensures fair comparison by using same graph state
- Provides detailed analysis and reporting
- Handles multiple scenarios efficiently

## 📈 Understanding Results

### Time Analysis
- **Positive time_saved**: Info sharing is faster
- **Negative time_saved**: Info sharing is slower
- **Zero time_saved**: No time difference

### Path Analysis
- **Positive paths_saved**: Info sharing explored fewer paths
- **Negative paths_saved**: Info sharing explored more paths
- **Zero paths_saved**: Same number of paths explored

### Efficiency Verdict
- **🏆 MORE EFFICIENT**: Info sharing saves both time and paths
- **⚠️ LESS EFFICIENT**: Info sharing costs time or paths
- **🤷 NEUTRAL**: No significant difference

## 🔧 Configuration Options

### Simulation Parameters
- `amount`: Payment amount in satoshis
- `p_willing`: Probability (0.0-1.0) that nodes share info
- `min_path_nodes`: Minimum nodes required in path
- `max_paths_checked`: Maximum paths to explore
- `include_endpoints_in_share`: Whether sender/receiver can share
- `seed_sender_receiver`: Fixed sender/receiver pair (None for random)
- `verbose`: Detailed output during simulation

### Example Scenarios
```python
# Small payment, low willingness
comparison.run_comparison(amount=10000.0, p_willing=0.3, min_path_nodes=2)

# Large payment, high willingness  
comparison.run_comparison(amount=100000.0, p_willing=0.7, min_path_nodes=4)

# Medium payment, medium willingness
comparison.run_comparison(amount=50000.0, p_willing=0.5, min_path_nodes=3)
```

## 🚀 Running the Code

### Basic Execution
```bash
cd /Users/manikantagudipudi/Desktop/CBS
source myenv/bin/activate
python network.py
```

### Example Script
```bash
python example_comparison.py
```

## 📋 Sample Output

```
================================================================================
LIGHTNING NETWORK ROUTING COMPARISON
================================================================================
Payment Amount: 70,000.0 satoshis
Willingness Probability: 60.0%
Min Path Nodes: 2
Max Paths Checked: 1000
Fixed Sender/Receiver: False
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
  Without Info Sharing: 0.1027 seconds
  With Info Sharing:    0.1556 seconds
  Time Saved:           -0.0529 seconds (-51.5%)

🛤️  PATHS CHECKED:
  Without Info Sharing: 1 paths
  With Info Sharing:    1 paths
  Paths Saved:          0 paths (+0.0%)

📈 EFFICIENCY ANALYSIS:
  ❌ Info sharing COST 0.0529s (51.5%)
  ⚖️  Same number of paths checked

🎯 VERDICT:
  ⚠️  Info sharing is LESS EFFICIENT!
================================================================================
```

## 🔍 Key Insights

1. **Info sharing overhead**: There's computational cost to sharing information
2. **Path pruning effectiveness**: Both modes use edge pruning to avoid re-trying failed paths
3. **Success rates**: Both modes can succeed, but efficiency varies
4. **Parameter sensitivity**: Results depend on payment amount, willingness, and path constraints

## 🛠️ Extending the Code

### Adding New Metrics
```python
@dataclass
class SimulationResult:
    # ... existing fields ...
    new_metric: Optional[float] = None
```

### Custom Comparison Logic
```python
def custom_comparison(self, result1, result2):
    # Your custom analysis
    return custom_metrics
```

### Different Sharing Strategies
```python
# Modify the willingness logic in SenderViewSim
def _custom_willingness(self, node, path):
    # Your custom willingness calculation
    return custom_probability
```

This enhanced simulation provides a robust framework for analyzing Lightning Network routing efficiency with and without information sharing!
