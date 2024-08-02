import re

def extract_number_from_brackets(s):
    match = re.search(r'\[(-?\d+)\]', s)
    if match:
        return int(match.group(1))
    return None