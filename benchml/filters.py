
def none(meta):
    return True

def regression_only(meta):
    return meta["task"] == "regression"

def classification_only(meta):
    return meta["task"] == "classification"

filters = {
    "none": none,
    "regression_only": regression_only,
    "classification_only": classification_only
}
