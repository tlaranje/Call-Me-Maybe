import argparse

""" TODO
Add a description to the parser
Add help descrition to each argument
"""


def get_args() -> dict[str, str]:
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--input',
        help="",
        default='data/input/function_calling_tests.json'
    )
    parser.add_argument(
        '--output',
        help="",
        default='data/output/function_calling_result.json'
    )
    parser.add_argument(
        "--functions_definitions",
        help="",
        default='data/input/functions_definition.json'
    )
    args = parser.parse_args()
    return {
        "Input": args.input,
        "Output": args.output,
        "Functions Definitions": args.functions_definitions
    }
