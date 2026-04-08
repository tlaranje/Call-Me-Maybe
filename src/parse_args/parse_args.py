import argparse


def parse_args() -> dict[str, str]:
    """
    Parses command-line arguments for the function-calling pipeline.

    Defines and processes CLI arguments for input prompts, output location,
    and function definitions.

    Returns:
        dict[str, str]: A dictionary containing normalized keys:
            - "input": Path to the input prompts file.
            - "output": Path to the output results file.
            - "functions": Path to the function definitions file.
    """
    parser = argparse.ArgumentParser(
        description="Function calling pipeline CLI"
    )

    parser.add_argument(
        "--input",
        help="Path to the input prompts JSON file.",
        default="data/input/function_calling_tests.json"
    )

    parser.add_argument(
        "--output",
        help="Path to the output JSON file.",
        default="data/output/function_calling_results.json"
    )

    parser.add_argument(
        "--functions_definition",
        help="Path to the functions definition JSON file.",
        default="data/input/functions_definition.json"
    )

    args = parser.parse_args()

    return {
        "input": args.input,
        "output": args.output,
        "functions": args.functions_definition
    }
