def serialize_logged_model(model) -> dict:
    d = model.to_dictionary()
    d["status"] = str(model.status)
    d["metrics"] = [m.to_dictionary() for m in (model.metrics or [])]
    return d
