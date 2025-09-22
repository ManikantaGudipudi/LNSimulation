import json
import sys
from collections import Counter
from datetime import datetime

def analyze_json_structure(data, max_depth=3, current_depth=0):
    """Recursively analyze JSON structure and return a summary"""
    if current_depth >= max_depth:
        return "..."
    
    if isinstance(data, dict):
        structure = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                structure[key] = analyze_json_structure(value, max_depth, current_depth + 1)
            else:
                structure[key] = type(value).__name__
        return structure
    elif isinstance(data, list):
        if len(data) > 0:
            return [analyze_json_structure(data[0], max_depth, current_depth + 1)]
        else:
            return []
    else:
        return type(data).__name__

def print_sample_data(data, sample_size=1):
    """Print sample data from the JSON"""
    if isinstance(data, dict):
        print("Sample dictionary entries:")
        for i, (key, value) in enumerate(data.items()):
            if i >= sample_size:
                break
            if isinstance(value, (dict, list)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")
    elif isinstance(data, list):
        print(f"Sample list entries (showing first {min(sample_size, len(data))}):")
        for i, item in enumerate(data[:sample_size]):
            if isinstance(item, dict):
                print(f"  [{i}]: Dictionary with keys: {list(item.keys())[:5]}{'...' if len(item) > 5 else ''}")
            else:
                print(f"  [{i}]: {item}")

def analyze_channels(channels):
    """Analyze channel data and provide statistics"""
    if not channels:
        return
    
    print(f"\n=== CHANNEL ANALYSIS ===")
    print(f"Total number of channels: {len(channels)}")
    
    # Analyze active channels
    active_channels = [ch for ch in channels if ch.get('active', False)]
    print(f"Active channels: {len(active_channels)} ({len(active_channels)/len(channels)*100:.1f}%)")
    
    # Analyze channel amounts
    amounts = [ch.get('satoshis', 0) for ch in channels if 'satoshis' in ch]
    if amounts:
        print(f"Channel amounts (satoshis):")
        print(f"  Min: {min(amounts):,}")
        print(f"  Max: {max(amounts):,}")
        print(f"  Average: {sum(amounts)/len(amounts):,.0f}")
    
    # Analyze fee structures
    base_fees = [ch.get('base_fee_millisatoshi', 0) for ch in channels if 'base_fee_millisatoshi' in ch]
    fee_rates = [ch.get('fee_per_millionth', 0) for ch in channels if 'fee_per_millionth' in ch]
    
    if base_fees:
        print(f"Base fees (millisatoshi):")
        print(f"  Min: {min(base_fees)}")
        print(f"  Max: {max(base_fees)}")
        print(f"  Average: {sum(base_fees)/len(base_fees):.1f}")
    
    if fee_rates:
        print(f"Fee rates (per millionth):")
        print(f"  Min: {min(fee_rates)}")
        print(f"  Max: {max(fee_rates)}")
        print(f"  Average: {sum(fee_rates)/len(fee_rates):.1f}")
    
    # Analyze delays
    delays = [ch.get('delay', 0) for ch in channels if 'delay' in ch]
    if delays:
        delay_counter = Counter(delays)
        print(f"Most common delays: {dict(delay_counter.most_common(5))}")

def print_detailed_structure(data, prefix="", max_items=3, max_depth=2, current_depth=0):
    """Print detailed structure showing actual keys and data types"""
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
        
    if isinstance(data, dict):
        print(f"{prefix}Dictionary with {len(data)} keys:")
        for i, (key, value) in enumerate(data.items()):
            if i >= max_items:
                print(f"{prefix}  ... and {len(data) - max_items} more keys")
                break
            print(f"{prefix}  '{key}': {type(value).__name__}")
            if isinstance(value, (dict, list)):
                print_detailed_structure(value, prefix + "    ", max_items=2, max_depth=max_depth, current_depth=current_depth+1)
    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items")
        if len(data) > 0:
            print(f"{prefix}  First item type: {type(data[0]).__name__}")
            if isinstance(data[0], (dict, list)):
                print_detailed_structure(data[0], prefix + "    ", max_items=2, max_depth=max_depth, current_depth=current_depth+1)
    else:
        sample_value = str(data)[:50] if len(str(data)) > 50 else str(data)
        print(f"{prefix}{type(data).__name__}: {sample_value}")

def main():
    json_file = "LNdata.json"
    
    try:
        print("Loading JSON file...")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print("✓ JSON file loaded successfully!")
        
        # Print detailed structure first
        print(f"\n=== DETAILED JSON STRUCTURE ===")
        print_detailed_structure(data)
        
        # Print overall structure
        print(f"\n=== JSON STRUCTURE SUMMARY ===")
        structure = analyze_json_structure(data)
        print(json.dumps(structure, indent=2))
        
        # Print sample data
        print(f"\n=== SAMPLE DATA ===")
        print_sample_data(data, 1)
        
        # If this is channel data, analyze it
        if isinstance(data, dict) and 'channels' in data:
            analyze_channels(data['channels'])
            
            # Show one example channel structure
            print(f"\n=== EXAMPLE CHANNEL STRUCTURE ===")
            if data['channels']:
                channel = data['channels'][0]
                print("First channel keys and types:")
                for key, value in channel.items():
                    print(f"  {key}: {type(value).__name__}")
                
                # Print complete first channel entry
                print(f"\n=== COMPLETE FIRST CHANNEL ENTRY ===")
                print(json.dumps(channel, indent=2))
        
        # File size information
        import os
        file_size = os.path.getsize(json_file)
        print(f"\n=== FILE INFORMATION ===")
        print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"Data type: {type(data).__name__}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
