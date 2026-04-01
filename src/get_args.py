import argparse


def get_args() -> dict[str, str]:
    """
    Parse command-line arguments and return resolved file paths.

    Defines CLI arguments for input prompts, output file, and
    function definitions, returning them in a structured dictionary.

    Args:
        None

    Returns:
        dict[str, str]: Dictionary containing:
            - "input": Path to input prompts file
            - "output": Path to output results file
            - "functions": Path to function definitions file
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Function calling pipeline CLI"
    )

    # Input prompts file
    parser.add_argument(
        "--input",
        help="Path to the input prompts JSON file.",
        default="data/input/function_calling_tests.json"
    )

    # Output results file
    parser.add_argument(
        "--output",
        help="Path to the output JSON file.",
        default="data/output/function_calling_results.json"
    )

    # Function definitions file
    parser.add_argument(
        "--functions_definition",
        help="Path to the functions definition JSON file.",
        default="data/input/functions_definition.json"
    )

    # Parse CLI arguments
    args = parser.parse_args()

    # Return normalized dictionary
    return {
        "input": args.input,
        "output": args.output,
        "functions": args.functions_definition
    }
