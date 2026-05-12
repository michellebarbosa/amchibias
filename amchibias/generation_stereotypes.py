import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load Secrets
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Configuration
INPUT_FILE = "data/raw/identity_groups.json"
OUTPUT_FILE = "data/raw/attributes_draft.json"
MODEL = "gpt-4o-mini"

def generate_attributes(group_name, axis):
    """
    Generates a list of stereotypical adjectives/attributes for a group.
    Based on Table 5 (Stereotype Gen) from the IndiBias paper.
    """
    prompt = f"""
    I am conducting academic research on sociolinguistic stereotypes in Goa (India).
    
    Target Group: {group_name}
    Axis: {axis}
    
    Task: List 10 common stereotypical adjectives or short phrases associated with '{group_name}' in the Goan context.
    - Include both positive and negative stereotypes.
    - Focus on local cultural nuances (e.g., specific to Goa).
    - Format: JSON list of strings only.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        # Handle cases where GPT wraps the list in a key like "adjectives" or "attributes"
        return result.get('adjectives', result.get('attributes', result.get('list', list(result.values())[0])))
    except Exception as e:
        print(f"    [Error] {group_name}: {e}")
        return []

def main():
    print(f" Starting Attribute Generation...\n")
    
    with open(INPUT_FILE, 'r') as f:
        identity_data = json.load(f)
        
    all_attributes = []
    
    for axis, groups in identity_data.items():
        print(f"Processing Axis: {axis}")
        
        for group in groups:
            print(f"  > Fetching attributes for: {group}")
            
            attrs = generate_attributes(group, axis)
            
            if attrs:
                # Add to our master list
                entry = {
                    "axis": axis,
                    "target_group": group,
                    "generated_attributes": attrs #require review by annotators
                }
                all_attributes.append(entry)
                print(f"    -> Found {len(attrs)} attributes")
            
            time.sleep(0.5)

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_attributes, f, indent=2)
        
    print("Attribute generation completed.")

if __name__ == "__main__":
    main()
