import math


def set_null_highest_token(logits):
    logits[logits.index(max(logits))] = -math.inf
    return logits
