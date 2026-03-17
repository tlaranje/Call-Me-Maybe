from llm_sdk import Small_LLM_Model as LLM_Model
from src.validation_models import FunctionDefinition as FuncDef
from src.utils import load_vocab
from typing import Any, Callable


def generate_number(
    model: LLM_Model, param_name: str, prompt: str
) -> dict[str, float]:
    return {"a": 2.0}


def generate_string(
        model: LLM_Model, param_name: str, prompt: str
) -> dict[str, str]:
    return {"name": "tlaranje"}


def generate_bool(
    model: LLM_Model, param_name: str, prompt: str
) -> dict[str, bool]:
    return {"is": True}


GeneratorFn = Callable[[LLM_Model, str, str], dict[str, Any]]

TYPE_GENERATORS: dict[str, GeneratorFn] = {
    "number": generate_number,
    "string": generate_string,
    "bool":   generate_bool,
}


def generate_parameters(
    model: LLM_Model, func: FuncDef, prompt: str
) -> dict[str, Any]:
    res = {}
    for param_name, param in func.parameters.items():
        generator = TYPE_GENERATORS.get(param.type)

        if generator is None:
            raise ValueError(
                f"No generator registered for type '{param.type}' "
                f"(parameter '{param_name}' of '{func.name}')"
            )

        res.update(generator(model, func.name, prompt))
    return res
