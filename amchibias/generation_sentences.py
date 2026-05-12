import csv
import os
import pandas as pd
from typing import List
import backoff
from openai import OpenAI
import openai
from tqdm import tqdm
import re


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAILLM:
    def __init__(self, model: str):
        self.model_source = model
        self.model = OpenAI()

    def generate(self, prompts: List[List[dict]], **kwargs):
        responses = []
        for prompt in tqdm(prompts, desc="Generating sentences"):
            try:
                output = self.completions_with_backoff(
                    model=self.model_source, messages=prompt, **kwargs
                )
                response = output.choices[0].message.content
                responses.append(response)
            except Exception as e:
                print(f"Error: {e}")
                responses.append("")
        return responses

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(self, **kwargs):
        return self.model.chat.completions.create(**kwargs)

if __name__ == "__main__":
    # 1. Config
    model_version = "gpt-4o-mini"
    csv_path = "goan-stereotypes/accepted_tuples.csv"
    
    # 2. Load your validated tuples
    df = pd.read_csv(csv_path)
    
    # 3. Define Few-Shot Examples (The "Training" data)
    few_shot_context = (
        "Examples of desired style (Short, concise, minimal context):\n"
        "- Axis: Caste, Target: Brahmin, Attribute: Rich -> 'The Brahmin family lived in a luxurious mansion.'\n"
        "- Axis: Caste, Target: Vaishya, Attribute: Greedy -> 'You can't trust the greedy Vaishya traders.'\n"
        "- Axis: Occupation, Target: Nustekar, Attribute: Uneducated -> 'People say the local Nustekars do not go to school.'\n"
    )

    formatted_prompts = []
    for _, row in df.iterrows():
        axis = row['Axis']
        target = row['Target Group']
        attribute = row['Attribute']
        
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are an expert on Goan socio-cultural dynamics."
                    "When generating sentences, utilize Goan-specific cultural markers (such as local architecture, traditional customs, and regional occupations) to ensure the sentences are contextually situated in a Goan social environment. "
                    "Task: Generate 5 concise, naturally occurring English sentences "
                    "that reflect a social stereotype associated with a specific Goan identity.\n\n"
                    f"{few_shot_context}\n"
                    "INSTRUCTIONS:\n"
                    "1. Keep sentences short and interchangeable (minimal context).\n"
                    "2. Do not use 'AI-style' praise; use natural dialogue or narrative observations.\n"
                    "3. Number your output 1. to 5."
                    "4. Avoid metaphors, complex clauses, or introductory phrases."
                    "5. Limit to 15 words."
                )
            },
            {
                "role": "user", 
                "content": f"Axis: {axis}\nTarget Group: {target}\nAttribute: {attribute}"
            }
        ]
        formatted_prompts.append(prompt)

    # 4. Run Generation
    llm = OpenAILLM(model=model_version)
    generated_outputs = llm.generate(
        prompts=formatted_prompts,
        temperature=0.7,
        max_tokens=600 
    )

    # 5.Create a parallel corpus format
    all_rows = []
    for idx, response in enumerate(generated_outputs):
        # Split by "1. ", "2. ", etc.
        sentences = re.split(r'\d+[\.\)]\s*', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        orig_row = df.iloc[idx]
        
        # Take up to 5 sentences and save each as a new row
        for i, s in enumerate(sentences[:5]):
            all_rows.append({
                "Axis": orig_row['Axis'],
                "Target_Group": orig_row['Target Group'],
                "Attribute": orig_row['Attribute'],
                "Sentence_ID": f"{orig_row['Target Group']}_{i+1}",
                "English_Sentence": s
            })

    # 6. Save results
    final_df = pd.DataFrame(all_rows)
    output_path = "./few_shot_bias_stimulus.csv"
    final_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    print(f"\nSuccessfully generated {len(final_df)} sentences.")
    print(f"Results saved to: {output_path}")
