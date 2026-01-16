# setup_chromadb_docker.py
# Quick setup script for ChromaDB Docker backend

import os

def update_env_file():
    """Update .env with ChromaDB Docker settings."""
    env_path = ".env"
    
    # Read current .env if exists
    current_lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            current_lines = f.readlines()
    
    # Settings to add/update
    docker_settings = {
        'VECTOR_STORE_TYPE': 'chromadb',
        'CHROMADB_HOST': 'localhost',
        'CHROMADB_PORT': '8000',
    }
    
    # Find and update existing settings, or mark for addition
    updated_keys = set()
    new_lines = []
    
    for line in current_lines:
        key = line.split('=')[0].strip() if '=' in line else None
        if key in docker_settings:
            new_lines.append(f"{key}={docker_settings[key]}\n")
            updated_keys.add(key)
        else:
            new_lines.append(line)
    
    # Add any missing settings
    for key, value in docker_settings.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}\n")
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    print("[OK] .env updated with ChromaDB Docker settings:")
    print("     VECTOR_STORE_TYPE=chromadb")
    print("     CHROMADB_HOST=localhost")
    print("     CHROMADB_PORT=8000")
    print("")
    print("ChromaDB Docker is ready to use!")


if __name__ == "__main__":
    print("=== ChromaDB Docker Setup ===\n")
    update_env_file()
