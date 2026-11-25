import ast

import sympy as sp


MAX_EXPRESSION_LENGTH = 500
MAX_AST_NODES = 128
MAX_POWER = 100
FUNCTIONS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
}
BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
}
UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def _validate_variable(variable):
    if (
        not isinstance(variable, str)
        or len(variable) > 21
        or not variable.isidentifier()
        or variable.startswith("_")
    ):
        raise ValueError("变量名必须是有效标识符，且不能以下划线开头")
    if variable in FUNCTIONS or variable in ("pi", "e", "E"):
        raise ValueError("变量名不能与内置函数或常数重名")
    return sp.Symbol(variable)


def _parse_node(node):
    if isinstance(node, ast.Expression):
        return _parse_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("表达式只支持数字常量")
        if abs(node.value) > 1e100:
            raise ValueError("数字常量过大")
        return sp.Integer(node.value) if isinstance(node.value, int) else sp.Float(node.value)
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sp.pi
        if node.id in ("e", "E"):
            return sp.E
        if (
            len(node.id) > 21
            or not node.id.isidentifier()
            or node.id.startswith("_")
        ):
            raise ValueError("表达式中包含不支持的名称")
        if node.id in FUNCTIONS:
            raise ValueError(f"{node.id} 必须作为函数调用")
        return sp.Symbol(node.id)
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY_OPERATORS:
        if isinstance(node.op, ast.Pow):
            exponent = node.right
            if (
                isinstance(exponent, ast.Constant)
                and isinstance(exponent.value, (int, float))
                and abs(exponent.value) > MAX_POWER
            ):
                raise ValueError(f"指数绝对值不能超过 {MAX_POWER}")
        return BINARY_OPERATORS[type(node.op)](
            _parse_node(node.left),
            _parse_node(node.right),
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY_OPERATORS:
        return UNARY_OPERATORS[type(node.op)](_parse_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function = FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError("表达式调用了不支持的函数")
        if node.func.id == "log":
            if len(node.args) not in (1, 2):
                raise ValueError("log 需要一个参数（或两个参数表示底数）")
        elif len(node.args) != 1:
            raise ValueError(f"{node.func.id} 需要一个参数")
        return function(*[_parse_node(argument) for argument in node.args])
    raise ValueError("表达式包含不支持的语法；仅接受算术、常见函数和变量")


def parse_expression(expression):
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("表达式不能为空")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"表达式长度不能超过 {MAX_EXPRESSION_LENGTH} 个字符")
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError(f"表达式复杂度不能超过 {MAX_AST_NODES} 个语法节点")
    return _parse_node(tree)


def _result(operation, expression, value, variable=None):
    unevaluated_types = (sp.Integral, sp.Derivative, sp.Limit, sp.ConditionSet)
    result = {
        "operation": operation,
        "input": expression,
        "result": str(value),
        "latex": sp.latex(value),
        "evaluated": not (
            value.has(*unevaluated_types)
            if isinstance(value, sp.Basic)
            else any(
                item.has(*unevaluated_types)
                for item in value
                if isinstance(item, sp.Basic)
            )
        ),
    }
    if variable is not None:
        result["variable"] = variable
    return result


def simplify(expression):
    return _result(
        "simplify",
        expression,
        sp.simplify(parse_expression(expression)),
    )


def differentiate(expression, variable="x", order=1):
    symbol = _validate_variable(variable)
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 10:
        raise ValueError("求导阶数必须是 1 到 10 之间的整数")
    value = sp.diff(parse_expression(expression), symbol, order)
    return _result("differentiate", expression, value, variable)


def integrate(expression, variable="x", lower=None, upper=None):
    symbol = _validate_variable(variable)
    if (lower is None) != (upper is None):
        raise ValueError("定积分必须同时提供 lower 和 upper")
    integrand = parse_expression(expression)
    if lower is None:
        value = sp.integrate(integrand, symbol)
    else:
        value = sp.integrate(
            integrand,
            (symbol, parse_expression(str(lower)), parse_expression(str(upper))),
        )
    return _result("integrate", expression, value, variable)


def solve_equation(expression, variable="x"):
    symbol = _validate_variable(variable)
    if expression.count("=") > 1:
        raise ValueError("方程最多只能包含一个等号")
    if "=" in expression:
        left, right = expression.split("=", 1)
        equation = parse_expression(left) - parse_expression(right)
    else:
        equation = parse_expression(expression)
    solutions = sp.solve(equation, symbol)
    return _result("solve_equation", expression, solutions, variable)


def calculate_limit(expression, variable="x", point="0", direction="+-"):
    symbol = _validate_variable(variable)
    if direction not in ("+", "-", "+-"):
        raise ValueError("direction 只能是 +、- 或 +-")
    value = sp.limit(
        parse_expression(expression),
        symbol,
        parse_expression(str(point)),
        dir=direction,
    )
    return _result("limit", expression, value, variable)
