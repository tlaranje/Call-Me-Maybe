import argparse


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
        "input": args.input,
        "output": args.output,
        "functions": args.functions_definitions
    }
