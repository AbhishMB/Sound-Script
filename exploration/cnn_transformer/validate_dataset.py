"""
Validate that your dataset files match the required format for Trascriptionmodel.py
"""
import numpy as np
from pathlib import Path


def validate_dataset(feature_file_list, label_file_list):
    """
    Validate dataset format requirements.
    
    Args:
        feature_file_list: List of paths to feature .npy files
        label_file_list: List of paths to label .npy files
    """
    print("=" * 60)
    print("Dataset Format Validation")
    print("=" * 60)
    
    # Check lists match
    if len(feature_file_list) != len(label_file_list):
        print(f"❌ ERROR: Mismatched list lengths!")
        print(f"   Features: {len(feature_file_list)} files")
        print(f"   Labels: {len(label_file_list)} files")
        return False
    
    print(f"✓ Found {len(feature_file_list)} file pairs\n")
    
    all_valid = True
    
    for i, (feat_path, label_path) in enumerate(zip(feature_file_list, label_file_list)):
        print(f"Checking pair {i+1}/{len(feature_file_list)}: {Path(feat_path).name}")
        
        try:
            # Load feature file
            feat = np.load(feat_path)
            
            # Validate feature shape
            if len(feat.shape) != 3:
                print(f"  ❌ Feature shape must be 3D, got {len(feat.shape)}D")
                all_valid = False
                continue
            
            if feat.shape[0] != 3:
                print(f"  ❌ Feature must have 3 channels, got {feat.shape[0]}")
                all_valid = False
                continue
            
            if feat.shape[1] != 128:
                print(f"  ❌ Feature must have 128 frequency bins, got {feat.shape[1]}")
                all_valid = False
                continue
            
            if feat.dtype != np.float32:
                print(f"  ⚠️  Warning: Feature dtype is {feat.dtype}, expected float32")
            
            # Load label file
            label = np.load(label_path)
            
            # Validate label shape
            if len(label.shape) != 2:
                print(f"  ❌ Label shape must be 2D, got {len(label.shape)}D")
                all_valid = False
                continue
            
            if label.shape[1] != 88:
                print(f"  ❌ Label must have 88 columns (piano keys), got {label.shape[1]}")
                all_valid = False
                continue
            
            if label.dtype != np.float32:
                print(f"  ⚠️  Warning: Label dtype is {label.dtype}, expected float32")
            
            # Check time alignment
            if feat.shape[2] != label.shape[0]:
                print(f"  ❌ Time frames mismatch!")
                print(f"     Feature: {feat.shape[2]} frames")
                print(f"     Label: {label.shape[0]} frames")
                all_valid = False
                continue
            
            # Check label values are binary
            unique_vals = np.unique(label)
            if not np.all(np.isin(unique_vals, [0.0, 1.0])):
                print(f"  ⚠️  Warning: Label contains non-binary values: {unique_vals}")
                print(f"     Expected only 0.0 and 1.0")
            
            # Check for NaN or Inf
            if np.any(np.isnan(feat)) or np.any(np.isinf(feat)):
                print(f"  ❌ Feature contains NaN or Inf values")
                all_valid = False
                continue
            
            if np.any(np.isnan(label)) or np.any(np.isinf(label)):
                print(f"  ❌ Label contains NaN or Inf values")
                all_valid = False
                continue
            
            print(f"  ✓ Valid - Feature: {feat.shape}, Label: {label.shape}")
            
        except FileNotFoundError as e:
            print(f"  ❌ File not found: {e}")
            all_valid = False
        except Exception as e:
            print(f"  ❌ Error loading files: {e}")
            all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ All files are valid! Dataset format is correct.")
    else:
        print("❌ Some files have errors. Please fix them before training.")
    print("=" * 60)
    
    return all_valid


if __name__ == "__main__":
    import sys
    import glob
    
    if len(sys.argv) < 3:
        print("Usage: python validate_dataset.py <features_dir> <labels_dir>")
        print("\nOr provide file lists directly:")
        print("  python validate_dataset.py --files")
        print("  Then edit the script to set feature_file_list and label_file_list")
        sys.exit(1)
    
    if sys.argv[1] == "--files":
        # User should edit these lists
        feature_file_list = []  # Add your feature file paths here
        label_file_list = []     # Add your label file paths here
        
        if len(feature_file_list) == 0:
            print("Please edit this script to set feature_file_list and label_file_list")
            sys.exit(1)
    else:
        # Auto-generate from directories
        features_dir = sys.argv[1]
        labels_dir = sys.argv[2]
        
        feature_file_list = sorted(glob.glob(f"{features_dir}/*_features.npy"))
        label_file_list = sorted(glob.glob(f"{labels_dir}/*_labels.npy"))
        
        if len(feature_file_list) == 0:
            print(f"No feature files found in {features_dir}")
            sys.exit(1)
        
        if len(label_file_list) == 0:
            print(f"No label files found in {labels_dir}")
            sys.exit(1)
    
    validate_dataset(feature_file_list, label_file_list)

