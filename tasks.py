def grade_easy(preds):
    gt = ["toxic"]
    return sum(p == g for p, g in zip(preds, gt)) / len(gt)


def grade_medium(preds):
    gt = ["toxic", "spam"]
    return sum(p == g for p, g in zip(preds, gt)) / len(gt)


def grade_hard(preds):
    gt = ["toxic", "spam", "safe"]
    score = sum(p == g for p, g in zip(preds, gt)) / len(gt)

    if score == 1.0:
        score = 0.95

    return score