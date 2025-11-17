#!/usr/bin/env python3
"""
Automatically place test files in the correct Unity test directory structure.
"""

import os
import shutil
from pathlib import Path

def get_test_location(component_name, test_type="Editor"):
    """Determine the correct test file location based on component name."""
    
    # VR-specific components
    vr_components = ["vr", "camera", "controller", "hand", "teleport", "spawn"]
    audio_components = ["audio", "sound", "brightness", "volume", "music"]
    gameplay_components = ["ant", "spider", "player", "game", "spawn", "ai"]
    ui_components = ["menu", "ui", "button", "dialog", "interface"]
    
    component_lower = component_name.lower()
    
    # Determine category
    if any(vr_comp in component_lower for vr_comp in vr_components):
        return f"Tests/{test_type}/VR/{component_name}Tests.cs"
    elif any(audio_comp in component_lower for audio_comp in audio_components):
        return f"Tests/{test_type}/Audio/{component_name}Tests.cs"
    elif any(gameplay_comp in component_lower for gameplay_comp in gameplay_components):
        return f"Tests/{test_type}/Gameplay/{component_name}Tests.cs"
    elif any(ui_comp in component_lower for ui_comp in ui_components):
        return f"Tests/{test_type}/UI/{component_name}Tests.cs"
    else:
        return f"Tests/{test_type}/{component_name}Tests.cs"

def create_test_directories(project_path):
    """Create the standard Unity test directory structure."""
    test_dirs = [
        "Tests",
        "Tests/Editor",
        "Tests/Editor/VR",
        "Tests/Editor/Audio", 
        "Tests/Editor/Gameplay",
        "Tests/Editor/UI",
        "Tests/Runtime"
    ]
    
    for dir_path in test_dirs:
        full_path = Path(project_path) / "Assets" / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {full_path}")

def place_test_file(project_path, component_name, test_content, test_type="Editor"):
    """Place a test file in the correct location."""
    
    # Get the correct location
    test_location = get_test_location(component_name, test_type)
    full_path = Path(project_path) / "Assets" / test_location
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the test file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"✅ Test file created: {full_path}")
    return str(full_path)

def main():
    """Main function to demonstrate test file placement."""
    print("🧪 Unity Test File Placement Helper")
    print("=" * 40)
    
    # Example usage
    project_path = "data/repos/11-VRapp_VR-project/The RegnAnt"
    
    print(f"\n📁 Creating test directories in: {project_path}")
    create_test_directories(project_path)
    
    print("\n🎯 Example test file placements:")
    examples = [
        ("BrightnessController", "Audio"),
        ("VRCamera", "VR"), 
        ("AntSpawner", "Gameplay"),
        ("MenuController", "UI"),
        ("PlayerMovement", "Gameplay")
    ]
    
    for component, category in examples:
        location = get_test_location(component)
        print(f"  {component} → {location}")

if __name__ == "__main__":
    main()

