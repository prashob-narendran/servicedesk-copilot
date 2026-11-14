from app.services.redact import Redactor


class FakeGliner:
    def predict_entities(self, text, labels, threshold=0.5):
        out = []
        for name, label in [("Anita Rao", "person"), ("Northwind Traders", "organization")]:
            i = text.find(name)
            if i >= 0:
                out.append({"start": i, "end": i + len(name), "text": name, "label": label, "score": 0.93})
        return out


def test_structured_pii_is_masked():
    text = ("Call me on +971 50 123 4567 or mail anita.rao@acme.com. Server 10.0.0.12 is down, "
            "see https://status.acme.com. Account number ACC-778812.")
    res = Redactor(None).redact(text)
    types = {e["type"] for e in res["entities"]}
    assert {"PHONE", "EMAIL", "IP_ADDRESS", "URL", "ACCOUNT_NUMBER"} <= types
    for e in res["entities"]:
        assert e["text"] not in res["redacted_text"]
    assert "[EMAIL_1]" in res["redacted_text"] and "[PHONE_1]" in res["redacted_text"]


def test_dates_and_small_numbers_are_not_phones():
    res = Redactor(None).redact("Since 2026-11-03 about 12 users and 5 printers fail; ticket 4471.")
    assert res["entities"] == []


def test_model_entities_people_masked_orgs_kept():
    text = "Hi, Anita Rao from Northwind Traders here. Anita Rao again: email anita@nw.com"
    res = Redactor(FakeGliner()).redact(text)
    persons = [e for e in res["entities"] if e["type"] == "PERSON"]
    assert persons and all(e["masked_as"] == "[PERSON_1]" for e in persons)
    assert "Northwind Traders" in res["redacted_text"]          # ORG reported but not masked
    assert "Anita Rao" not in res["redacted_text"]
    org = next(e for e in res["entities"] if e["type"] == "ORG")
    assert org["masked_as"] is None


def test_offsets_point_to_original_text():
    text = "Email me at x.y@z.org please"
    e = Redactor(None).redact(text)["entities"][0]
    assert text[e["start"]:e["end"]] == e["text"]
