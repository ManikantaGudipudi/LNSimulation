# Lightning Network Pair Testing Guide

## 🎯 Overview

The updated Lightning Network simulation now allows you to easily test multiple sender/receiver pairs without changing seeds. You can specify exact pairs or generate random ones.

## 🚀 Quick Start

### 1. Test Random Pairs (Default)
```bash
python network.py
```
This will automatically generate 3 random pairs and test them.

### 2. Test Specific Pairs
Edit the `main()` function in `network.py`:

```python
def main():
    # Define your specific pairs here
    sender_receiver_pairs = [
        ("node_id_1", "node_id_2"),
        ("node_id_3", "node_id_4"),
        ("node_id_5", "node_id_6"),
    ]
    
    # Rest of the code...
```

### 3. Use the Example Script
```bash
python test_pairs.py
```

## 📊 What You Get

### Individual Pair Results
For each pair, you'll see:
- **Success Rate**: Whether both modes succeeded
- **Execution Time**: Time taken by each mode
- **Time Saved**: Difference in execution time
- **Paths Checked**: Number of paths explored
- **Efficiency Verdict**: Whether info sharing helped or hurt

### Overall Summary
- **Successful pairs**: How many pairs succeeded in both modes
- **Average time saved**: Average time benefit from info sharing
- **Average paths saved**: Average path exploration benefit

## 🔧 Customization Options

### In `network.py` main function:
```python
# Test specific pairs
sender_receiver_pairs = [
    ("your_sender_1", "your_receiver_1"),
    ("your_sender_2", "your_receiver_2"),
]

# Or generate random pairs
sender_receiver_pairs = get_random_sender_receiver_pairs(
    json_file="LNdata.json", 
    num_pairs=5,  # Number of random pairs
    seed=42       # For reproducibility
)

# Customize simulation parameters
result = comparison.run_comparison(
    sender=sender,
    receiver=receiver,
    amount=50000.0,               # Payment amount
    p_willing=0.7,                # Willingness to share (70%)
    min_path_nodes=3,             # Minimum path length
    max_paths_checked=500,        # Maximum paths to check
    verbose=True                  # Detailed output
)
```

### In `test_pairs.py`:
```python
def test_specific_pairs():
    # Add your specific pairs here
    sender_receiver_pairs = [
        ("node_id_1", "node_id_2"),
        ("node_id_3", "node_id_4"),
    ]
    
    # Test different scenarios
    scenarios = [
        {"amount": 10000.0, "p_willing": 0.3, "name": "Small payment, low willingness"},
        {"amount": 100000.0, "p_willing": 0.8, "name": "Large payment, high willingness"},
    ]
```

## 📈 Sample Output

```
====================================================================================================
PAIR 1/3: 03ac3c... -> 03c6d1...
====================================================================================================
================================================================================
LIGHTNING NETWORK ROUTING COMPARISON
================================================================================
Sender: 03ac3c0544fb9eee3977e4fb34a4b29276eb6a594c6fd9a7a6c7d121d7de71643d
Receiver: 03c6d1cce30eaaa8b4ad436735c6816ae886f588ff8362017af940ad5a89118014
Payment Amount: 70,000.0 satoshis
Willingness Probability: 60.0%
Min Path Nodes: 2
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
  Without Info Sharing: 0.0995 seconds
  With Info Sharing:    0.0999 seconds
  Time Saved:           -0.0004 seconds (-0.4%)

🛤️  PATHS CHECKED:
  Without Info Sharing: 1 paths
  With Info Sharing:    1 paths
  Paths Saved:          0 paths (+0.0%)

📈 EFFICIENCY ANALYSIS:
  ❌ Info sharing COST 0.0004s (0.4%)
  ⚖️  Same number of paths checked

🎯 VERDICT:
  ⚠️  Info sharing is LESS EFFICIENT!
================================================================================

====================================================================================================
SUMMARY OF ALL PAIRS
====================================================================================================
✅ Pair 1: Time saved: -0.0004s (-0.4%), Paths saved: 0 (+0.0%)
❌ Pair 2: Time saved: 0.0351s (+25.9%), Paths saved: 0 (+0.0%)
✅ Pair 3: Time saved: 0.0403s (+28.2%), Paths saved: 0 (+0.0%)

📊 OVERALL STATISTICS:
  Successful pairs: 2/3
  Average time saved: 0.0200s
  Average paths saved: 0.0
```

## 🎯 Key Benefits

1. **No More Seed Changing**: Test multiple pairs without modifying code
2. **Easy Pair Management**: Simply add pairs to a list
3. **Random Generation**: Automatically generate test pairs
4. **Comprehensive Analysis**: See results for each pair and overall statistics
5. **Flexible Parameters**: Customize payment amounts, willingness, etc.

## 🔍 Understanding Results

- **✅ SUCCESS**: Both modes found a route
- **❌ FAILED**: One or both modes failed to find a route
- **Time Saved**: Positive = info sharing faster, Negative = info sharing slower
- **Paths Saved**: Positive = info sharing checked fewer paths
- **Verdict**: Overall efficiency assessment

## 🛠️ Advanced Usage

### Testing Multiple Scenarios
```python
# Test same pairs with different parameters
scenarios = [
    {"amount": 10000.0, "p_willing": 0.3},
    {"amount": 50000.0, "p_willing": 0.6},
    {"amount": 100000.0, "p_willing": 0.9},
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

### Batch Testing
```python
# Test many random pairs
pairs = get_random_sender_receiver_pairs(num_pairs=10, seed=42)
for sender, receiver in pairs:
    result = comparison.run_comparison(sender, receiver, ...)
```

This makes it much easier to test different sender/receiver combinations and analyze the effectiveness of information sharing across various scenarios! 🎉
