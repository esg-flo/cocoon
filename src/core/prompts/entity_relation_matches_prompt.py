prompt = """Your goal is to build relations between input and reference entities.
The input entity has the following attributes:
{}
Below are reference entities:
{}
Do the following:
1. Read input entity attributes and guess what it is about.

2. Go through each output entity. Describe what it is and reason its relation.
For instance, given the input entity "small car":
if the same entity then EXACT_MATCH.
E.g., "small automobile"
else if has assumptions that is clearly wrong then CONFLICTED_ASSUMPTION
E.g., "big car" is wrong because input entity clearly specify size as "small"
else if additional assumptions that can't be verified then ADDITIONAL_ASSUMPTION
E.g., "electronic car" is additional battery assumption can't be verified
else if it is general super class then GENERAL
E.g., "small vehicle" and "car" are general category of "small car"
else it is irrelavent entity then NOT_RELATED
E.g., "cloth" is a irrelavent

Provide your answer as json:
```json
{{
    "Input Entity Guess": "...",
    "EXACT_MATCH": {{
        "reason": "The input entity is ... which matches ...",
        "entity": [...] (a list of index numbers. Note that each entity appears only once in one of the categories)
    }},
    "CONFLICTED_ASSUMPTION": {{
        "reason": "The (what specific) details are conflicted",
        "entity": [...]
    }},
    "ADDITIONAL_ASSUMPTION": {{
        "reason": "The (what specific) details are not mentioned",
        "entity": [...]
    }},
    "GENERAL": {{
        "reason": "...",
        "entity": [...],
    }},
    "NOT_RELATED": {{
        "reason": "...",
        "entity": [...],
    }},
    "Summary of Relations": "The input entity is ... (desrcibe its properties) It doesn't make assumptions about ... It is different from ..."
}}
```"""
