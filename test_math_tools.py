import unittest

import math_tools
import tool_registry


class MathToolsTests(unittest.TestCase):
    def test_symbolic_calculus_operations(self):
        self.assertEqual(
            math_tools.differentiate("x^3 + sin(x)")["result"],
            "3*x**2 + cos(x)",
        )
        self.assertEqual(math_tools.integrate("2*x")["result"], "x**2")
        self.assertEqual(math_tools.solve_equation("x^2=1")["result"], "[-1, 1]")
        self.assertEqual(
            math_tools.calculate_limit("sin(x)/x")["result"],
            "1",
        )
        self.assertTrue(math_tools.integrate("2*x")["evaluated"])

    def test_parser_rejects_executable_syntax(self):
        with self.assertRaises((ValueError, SyntaxError)):
            math_tools.simplify("__import__('os').system('whoami')")

    def test_registered_math_tool_returns_machine_readable_result(self):
        result = tool_registry.execute_tool(
            "math.differentiate",
            {"expression": "x^2"},
        )
        self.assertIn('"result": "2*x"', result)


if __name__ == "__main__":
    unittest.main()
