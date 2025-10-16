import os
import re
import ast
import yaml
from yaml.loader import SafeLoader

class TupleLoader(SafeLoader):
    pass

def parse_range(s):
    """
    Convert string with brackets to tuple
    Allow percentage, E-notation, signed and decimal format
    Support both parentheses () and square brackets []
    
    Example:
    '(1, 5)' -> (1, 5)
    '[1, 5]' -> (1, 5)
    '(10%, 25%)' -> (0.1, 0.25)
    '[10%, 25%]' -> (0.1, 0.25)
    """

    pattern = r'([-+]?\d+\.?\d*e[-+]?\d+|[-+]?\d+\.?\d*|[-+]?\.\d+)%'
    
    # replace all percentage to decimal
    def replace_percent(match):
        value = float(match.group(1))
        return str(value / 100)
    
    processed = re.sub(pattern, replace_percent, s)
    
    return ast.literal_eval(processed)

def list_constructor(loader, node):
    """Handle list format and convert to tuple with percentage processing"""
    value = loader.construct_sequence(node)
    
    # Check if it's a two-element list that could be a range
    if len(value) == 2:
        # Convert list to string format for processing
        str_value = f"[{value[0]}, {value[1]}]"
        try:
            return parse_range(str_value)
        except:
            return value
    return value

def tuple_constructor(loader, node):
    value = loader.construct_scalar(node)
    # 支持小括号和中括号两种格式
    pattern = r'^[\(\[]\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?%?)\s*,\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?%?)\s*[\)\]]$'
    if re.match(pattern, value):
        return parse_range(value)
    return value

TupleLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG,
    tuple_constructor
)

TupleLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG,
    list_constructor
)

class yamler:
    
    def __init__(self, yaml_path):
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f'File not exist, please check if path is correct: {yaml_path}')
        self.yaml_path = yaml_path

    def get_yaml_fields(self, fields):
        with open(self.yaml_path, 'r') as file:
            yaml_data = yaml.load(file, Loader=yaml.FullLoader)
            result = []
            for field in fields:
                result.append(yaml_data.get(field))
        return result

    def get_all_safe_fields(self):
        with open(self.yaml_path, 'r') as file:
            return yaml.safe_load(file.read())

    def get_all_fields(self):
        with open(self.yaml_path, 'r', encoding='utf-8-sig') as file:
            return yaml.load(file.read(), Loader=TupleLoader)
