import yaml
import os
from pathlib import Path


def fix_data_yaml(yaml_path):
    """
    Fix paths in data.yaml file using absolute paths
    """
    # Get absolute path to the yaml file's directory
    base_dir = os.path.abspath(os.path.dirname(yaml_path))

    # Create absolute paths for each directory
    train_path = os.path.join(base_dir, 'train', 'images')
    valid_path = os.path.join(base_dir, 'valid', 'images')
    test_path = os.path.join(base_dir, 'test', 'images')

    # Create new yaml content with absolute paths
    data = {
        'train': train_path,
        'val': valid_path,
        'test': test_path,
        'nc': 1,
        'names': ['empty-seat-frontal']
    }

    # Save the corrected yaml file
    with open(yaml_path, 'w') as file:
        yaml.dump(data, file, sort_keys=False)

    # Verify paths
    print("\nVerifying paths:")
    for path_type, path in [('Train', train_path), ('Valid', valid_path), ('Test', test_path)]:
        if os.path.exists(path):
            print(f"✓ {path_type} path exists: {path}")
            # Count images in directory
            image_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  Found {image_count} images")
        else:
            print(f"✗ {path_type} path missing: {path}")


if __name__ == "__main__":
    yaml_path = "data/data.yaml"
    fix_data_yaml(yaml_path)