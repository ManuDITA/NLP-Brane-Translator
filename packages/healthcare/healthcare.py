#!/usr/bin/env python3
import os
import yaml

def analyze_heart_disease(dataset: str) -> str:
    # Placeholder analysis — replace with real logic later
    result = f"Analyzed dataset: {dataset}. Found 3 risk factors."
    return result

if __name__ == "__main__":
    # Brane passes inputs via environment variables
    dataset = os.environ.get("dataset", "heart-disease")
    
    result = analyze_heart_disease(dataset)
    
    # Brane reads output as YAML — must print this exact format
    print(yaml.dump({"output": result}))