#!/usr/bin/env python3
"""
Example script showing how to test specific sender/receiver pairs.
You can easily modify the pairs list to test different combinations.
"""

from network import SimulationComparison, get_random_sender_receiver_pairs

def test_specific_pairs():
    """Test specific sender/receiver pairs."""
    print("🎯 Testing specific sender/receiver pairs...")
    
    # Define your specific pairs here
    sender_receiver_pairs = [
        # Add your pairs here - replace with actual node IDs from your data
        # ("node_id_1", "node_id_2"),
        # ("node_id_3", "node_id_4"),
        # ("node_id_5", "node_id_6"),
    ]
    
    # If no specific pairs, generate some random ones
    if not sender_receiver_pairs:
        print("No specific pairs provided, generating random pairs...")
        sender_receiver_pairs = get_random_sender_receiver_pairs(
            json_file="LNdata.json", 
            num_pairs=3, 
            seed=42
        )
    
    # Run comparisons
    comparison = SimulationComparison(json_file="LNdata.json", seed=42)
    
    for i, (sender, receiver) in enumerate(sender_receiver_pairs, 1):
        print(f"\n{'='*80}")
        print(f"PAIR {i}: {sender} -> {receiver}")
        print(f"{'='*80}")
        
        result = comparison.run_comparison(
            sender=sender,
            receiver=receiver,
            amount=50000.0,               # payment amount
            p_willing=0.7,                # willingness to share info
            min_path_nodes=2,             # minimum path length
            max_paths_checked=500,        # maximum paths to check
            verbose=False                 # set to True for detailed output
        )
        
        # Print quick summary
        print(f"\n📊 Quick Summary:")
        print(f"  Time saved: {result.time_saved:.4f}s ({result.time_saved_percentage:+.1f}%)")
        print(f"  Paths saved: {result.paths_saved} ({result.paths_saved_percentage:+.1f}%)")
        print(f"  Both successful: {result.no_sharing.success and result.with_sharing.success}")

def test_multiple_scenarios():
    """Test different scenarios with the same pairs."""
    print("\n🔄 Testing multiple scenarios with same pairs...")
    
    # Get some random pairs
    pairs = get_random_sender_receiver_pairs(num_pairs=2, seed=42)
    
    scenarios = [
        {"amount": 10000.0, "p_willing": 0.3, "name": "Small payment, low willingness"},
        {"amount": 100000.0, "p_willing": 0.8, "name": "Large payment, high willingness"},
    ]
    
    comparison = SimulationComparison(json_file="LNdata.json", seed=42)
    
    for sender, receiver in pairs:
        print(f"\n{'='*80}")
        print(f"Testing pair: {sender} -> {receiver}")
        print(f"{'='*80}")
        
        for scenario in scenarios:
            print(f"\n--- {scenario['name']} ---")
            result = comparison.run_comparison(
                sender=sender,
                receiver=receiver,
                amount=scenario["amount"],
                p_willing=scenario["p_willing"],
                min_path_nodes=2,
                max_paths_checked=200,
                verbose=False
            )
            
            print(f"  Time saved: {result.time_saved:.4f}s ({result.time_saved_percentage:+.1f}%)")
            print(f"  Paths saved: {result.paths_saved} ({result.paths_saved_percentage:+.1f}%)")

if __name__ == "__main__":
    print("🚀 Lightning Network Pair Testing")
    print("="*50)
    
    # Test specific pairs
    test_specific_pairs()
    
    # Test multiple scenarios
    test_multiple_scenarios()
    
    print("\n✅ All tests completed!")
