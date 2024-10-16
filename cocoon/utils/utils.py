import ast
import re


def parse_json_col(df, col: str):
    df[col] = df[col].apply(ast.literal_eval)
    return df


def _extract_json_code(s: str):
    pattern = r"```json(.*?)```"
    match = re.search(pattern, s, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_json_code_safe(s: str):
    s_stripped = s.strip()
    if (s_stripped.startswith("{") and s_stripped.endswith("}")) or (
        s_stripped.startswith("[") and s_stripped.endswith("]")
    ):
        return s_stripped
    return _extract_json_code(s_stripped)
