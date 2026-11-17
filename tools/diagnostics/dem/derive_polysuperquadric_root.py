"""Generate symbolic ray/root derivatives for implicit DEM particles.

This is a developer diagnostic, not a pytest test.  It has no import-time
side effects and prints the requested SymPy expression only when executed.
"""

import argparse


def build_expression(kind):
    try:
        import sympy as sp
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "this symbolic diagnostic requires the optional 'sympy' package"
        ) from error

    t = sp.Symbol("t", real=True)
    velocity = sp.Matrix(sp.symbols("v0:3", real=True))
    origin = sp.Matrix(sp.symbols("p0:3", real=True))
    radius = sp.Matrix(sp.symbols("r0:3", positive=True))
    point = origin + t * velocity

    if kind == "ellipsoid":
        epsilon_e, epsilon_n = sp.symbols(
            "epsilon_e epsilon_n", positive=True
        )
        xy = (
            sp.Abs(point[0] / radius[0]) ** (2 / epsilon_e)
            + sp.Abs(point[1] / radius[1]) ** (2 / epsilon_e)
        ) ** (epsilon_e / epsilon_n)
        expression = (
            xy + sp.Abs(point[2] / radius[2]) ** (2 / epsilon_n) - 1
        )
    else:
        epsilon = sp.Matrix(
            sp.symbols("epsilon_x epsilon_y epsilon_z", positive=True)
        )
        expression = sum(
            sp.Abs(point[index] / radius[index]) ** (2 / epsilon[index])
            for index in range(3)
        ) - 1

    return t, expression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("ellipsoid", "quadrics"), default="quadrics"
    )
    parser.add_argument(
        "--order",
        choices=("value", "gradient", "hessian"),
        default="gradient",
    )
    args = parser.parse_args()

    try:
        import sympy as sp
    except ModuleNotFoundError:
        parser.error(
            "this symbolic diagnostic requires the optional 'sympy' package"
        )

    parameter, expression = build_expression(args.kind)
    if args.order == "gradient":
        expression = sp.diff(expression, parameter)
    elif args.order == "hessian":
        expression = sp.diff(expression, parameter, 2)

    replacements, reduced = sp.cse(expression)
    for symbol, value in replacements:
        print(f"{symbol} = {sp.pycode(value)}")
    print(sp.pycode(reduced[0]))


if __name__ == "__main__":
    main()
