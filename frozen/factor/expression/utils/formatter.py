import re
import ast
import inspect
from .. import operators
from ...utils import preprocess

_ops_names = None

def _get_ops_names():
    global _ops_names
    if _ops_names is None:
        _ops_names = [name for name, obj in inspect.getmembers(operators) if inspect.isfunction(obj) and obj.__module__ == operators.__name__]
        _prep_names = [name for name, obj in inspect.getmembers(preprocess) if inspect.isfunction(obj) and obj.__module__ == preprocess.__name__]
        _method_names = [method for method in operators.Factor.__dict__.keys() if not(method.startswith('__') or method.endswith('__'))]
        _ops_names.extend(_prep_names)
        _ops_names.extend(_method_names)
    return _ops_names


def _replace_ops(match):
    _ops_names = _get_ops_names()
    group = match.group()
    return group.lower() if group.lower() in _ops_names else group


# class ReplaceComparisonTransformer(ast.NodeTransformer):

#     def visit_Compare(self, node):

#         for op in node.ops:

#             if isinstance(op, ast.Lt):
#                 new_node = ast.Call(
#                     func=ast.Name(id='lt', ctx=ast.Load()),
#                     args=[node.left, node.comparators[0]],
#                     keywords=[]
#                 )
#             elif isinstance(op, ast.Gt):
#                 new_node = ast.Call(
#                     func=ast.Name(id='gt', ctx=ast.Load()),
#                     args=[node.left, node.comparators[0]],
#                     keywords=[]
#                 )
#             elif isinstance(op, ast.LtE):
#                 new_node = ast.Call(
#                     func=ast.Name(id='lte', ctx=ast.Load()),
#                     args=[node.left, node.comparators[0]],
#                     keywords=[]
#                 )
#             elif isinstance(op, ast.GtE):
#                 new_node = ast.Call(
#                     func=ast.Name(id='gte', ctx=ast.Load()),
#                     args=[node.left, node.comparators[0]],
#                     keywords=[]
#                 )
#             elif isinstance(op, ast.Eq):
#                 new_node = ast.Call(
#                     func=ast.Name(id='eq', ctx=ast.Load()),
#                     args=[node.left, node.comparators[0]],
#                     keywords=[]
#                 )
#             elif isinstance(op, ast.NotEq):
#                 new_node = ast.Call(
#                     func=ast.Name(id='ne', ctx=ast.Load()),
#                     args=[node.left, node.comparators[0]],
#                     keywords=[]
#                 )
#             return new_node
        
#         return node


# transformer = ReplaceComparisonTransformer()


# def replace_comparison_ops(original_string):

#     # analyse original string
#     parsed_expression = ast.parse(original_string, mode='eval')
#     # replace ops with user-defined NodeTransformer
#     transformed_expression = transformer.visit(parsed_expression)
#     result_string = ast.unparse(transformed_expression)

#     return result_string


def str2expr(string):

    lower_str = re.sub(r'\b[A-Za-z_]+[A-Za-z]\b', _replace_ops, string)
    expr = re.sub(r'[?:]', ',', lower_str)
    # expr = replace_comparison_ops(expr)

    return expr

