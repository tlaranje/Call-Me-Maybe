import argparse


def get_args() -> dict[str, str]:
    """
    Parse command-line arguments and return the resolved file paths.

    Args:
        None

    Returns:
        dict[str, str]: A dictionary containing the input, output,
        and functions definition file paths.
    """
    parser = argparse.ArgumentParser(description='')

    parser.add_argument(
        '--input',
        help="Path to the input prompts JSON file.",
        default='data/input/function_calling_tests.json'
    )
    parser.add_argument(
        '--output',
        help="Path to the output JSON file.",
        default='data/output/function_calling_results.json'
    )
    parser.add_argument(
        "--functions_definition",
        help="Path to the functions definition JSON file.",
        default='data/input/functions_definition.json'
    )

    args = parser.parse_args()

    return {
        "input": args.input,
        "output": args.output,
        "functions": args.functions_definition
    }
