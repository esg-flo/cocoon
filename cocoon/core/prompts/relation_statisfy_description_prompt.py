prompt = """Find entities that satisfy the description:
Below are entities:
{}
Below are the description:
{}

Now, find entities that satisfy the description. Provide your answer as json:
```json
{{
    "reasoning": "The entities are about ...",
    "indices": [...] (list of index numbers, could be empty)
}}
``` """
