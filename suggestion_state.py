latest_suggestion = None
latest_image_b64 = None


def set_suggestion(data: dict, image_b64: str = None):
    global latest_suggestion, latest_image_b64
    latest_suggestion = data
    latest_image_b64 = image_b64


def get_suggestion():
    return latest_suggestion, latest_image_b64


def clear_suggestion():
    global latest_suggestion, latest_image_b64
    latest_suggestion = None
    latest_image_b64 = None
