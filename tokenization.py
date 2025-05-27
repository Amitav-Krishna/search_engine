import re

def tokenization(string, lowercase=True, keep_apostrophes=True, split_hyphens=False):
    if lowercase:
            string = string.lower()
    
    if keep_apostrophes:
        string = re.sub(r"'\w\b", "", string)  
    
    tokens = []
    if split_hyphens:
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", string)
    else:
        tokens = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", string)
    
    tokens = [t for t in tokens if t and t != "'"]
    
    return tokens