def dataset_descriptor(card_record, node_name):
    base = node_name.replace("/", " ").replace("_", " ").replace("-", " ")
    if not card_record:
        return base
    card_data = card_record.get("cardData") or {}
    tasks = card_data.get("task_categories") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    description = (card_record.get("description") or "")[:400]
    tags = [tag for tag in (card_record.get("tags") or []) if ":" not in tag][:10]
    return " ".join(part for part in (
        base,
        " ".join(map(str, tasks)),
        " ".join(tags),
        description,
    ) if part)
