from scale1m.dataset_descriptor import dataset_descriptor


def test_dataset_descriptor_name_only_fallback():
    assert dataset_descriptor(None, "owner/my_dataset-v2") == "owner my dataset v2"


def test_dataset_descriptor_preserves_frozen_field_order_and_filters_namespaced_tags():
    record = {
        "cardData": {"task_categories": "text-classification"},
        "tags": ["language:en", "classification", "small"],
        "description": "A short description.",
    }
    assert dataset_descriptor(record, "owner/data") == (
        "owner data text-classification classification small A short description."
    )
