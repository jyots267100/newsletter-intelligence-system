import pytest
from pydantic import ValidationError

from app.schemas.analysis import ActionItem, SourceType
from app.schemas.common import Citation


def test_citation_requires_page_number_at_least_one():
    with pytest.raises(ValidationError):
        Citation(document_id="d", filename="n.pdf", page_number=0, chunk_id="c1")

    citation = Citation(document_id="d", filename="n.pdf", page_number=1, chunk_id="c1")
    assert citation.page_number == 1


@pytest.mark.parametrize("placeholder", ["unknown", "Unknown", "N/A", "TBD", "not specified", ""])
def test_action_item_nulls_out_placeholder_owner_and_deadline(placeholder):
    item = ActionItem(action="Do the thing.", owner=placeholder, deadline=placeholder)
    assert item.owner is None
    assert item.deadline is None


def test_action_item_keeps_real_owner_and_deadline():
    item = ActionItem(action="Ship it.", owner="Maria Chen", deadline="Nov 15, 2026")
    assert item.owner == "Maria Chen"
    assert item.deadline == "Nov 15, 2026"


def test_action_item_defaults_to_explicit_source_type():
    item = ActionItem(action="Do the thing.")
    assert item.source_type == SourceType.EXPLICIT
