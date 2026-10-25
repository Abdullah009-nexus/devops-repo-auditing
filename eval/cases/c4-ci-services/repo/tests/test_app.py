from app import total


def test_total_rounds_to_cents():
    assert total([1.005, 2.0]) == 3.0
