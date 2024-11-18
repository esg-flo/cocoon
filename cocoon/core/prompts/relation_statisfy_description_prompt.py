prompt = """Find entities that satisfy the description:
Below are entities:
{}
Below are the description:
{}

Now, find entities that satisfy the description. Provide your answer as RFC8259 compliant JSON response following the output format without deviation.:```\n
{{
    "reasoning": "The entities are about ...",
    "indices": [...] (list of index numbers, could be empty)
}}
``` """
