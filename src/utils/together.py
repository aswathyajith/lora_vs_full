## Functions for TogetherAI

import os
from together import Together

class TogetherAPI:
    def __init__(self):
        self.api_key = os.getenv("TOGETHER_API_KEY")
        self.client = Together(api_key=self.api_key)

    def list_all_models(self):
        models = self.together.models.list()
        return [model.id for model in models]

    def prompt_model(
        self, 
        model_name, 
        prompt, 
        max_tokens=100,
        temperature=0):
        client = self.client
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            stream=False, 
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].text

    def finetune_model(self, model_name, dataset_path, output_path):
        model = self.together.models.fine_tune(model_name, dataset_path, output_path)
        return model