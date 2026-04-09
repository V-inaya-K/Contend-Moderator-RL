# def grade_easy(preds):
#     gt = ["toxic"]
#     return sum(p == g for p, g in zip(preds, gt)) / len(gt)


# def grade_medium(preds):
#     gt = ["toxic", "spam"]
#     return sum(p == g for p, g in zip(preds, gt)) / len(gt)


# def grade_hard(preds):
#     gt = ["toxic", "spam", "safe"]
#     score = sum(p == g for p, g in zip(preds, gt)) / len(gt)

#     if score == 1.0:
#         score = 0.95

#     return score

# ------------

from typing import List

def grade_easy(preds: List[str]) -> float:
    gt = ["safe"]
    return sum(p == g for p, g in zip(preds, gt)) / len(gt)

def grade_medium(preds: List[str]) -> float:
    gt = ["toxic", "spam"]
    return sum(p == g for p, g in zip(preds, gt)) / len(gt)


def grade_hard(preds: List[str]) -> float:
    gt = ["toxic", "spam", "safe"]
    score = sum(p == g for p, g in zip(preds, gt)) / len(gt)

    if score == 1.0:
        score = 0.95
    return score

TASKS = {
    "easy": {
        "grader": grade_easy
    },
    "medium": {
        "grader": grade_medium
    },
    "hard": {
        "grader": grade_hard
    }
}
