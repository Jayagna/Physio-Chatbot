import pytest

from physiobot.safety import check_red_flags, urgent_reply


@pytest.mark.parametrize(
    "message, flag",
    [
        ("I have chest pain going to my left arm", "possible heart problem"),
        ("My face is drooping and I have slurred speech", "possible stroke"),
        ("Suddenly weakness in my right arm and leg", "possible stroke"),
        ("Back pain and I've lost control of my bladder", "possible spinal cord compression (cauda equina)"),
        ("numbness between my legs with back ache", "possible spinal cord compression (cauda equina)"),
        ("I fell and now I can't put weight on my ankle", "possible fracture or serious injury"),
        ("My calf is swollen and painful after knee surgery", "possible blood clot"),
        ("I have fever and back pain since 3 days", "possible infection"),
        ("I want to end my life", "self-harm"),
        ("छाती में दर्द हो रहा है", "possible heart problem"),
        ("છાતીમાં દુખાવો થાય છે", "possible heart problem"),
    ],
)
def test_detects_red_flags(message, flag):
    result = check_red_flags(message)
    assert result is not None
    assert result.name == flag


@pytest.mark.parametrize(
    "message",
    [
        "What exercises help lower back pain?",
        "How long does recovery from ACL surgery take?",
        "My knee hurts when I climb stairs",
        "What are the clinic timings?",
        "ઘૂંટણના દુખાવા માટે શું કરવું?",
        "Can physiotherapy help with frozen shoulder?",
    ],
)
def test_ordinary_questions_pass(message):
    assert check_red_flags(message) is None


def test_urgent_reply_mentions_emergency_number():
    flag = check_red_flags("crushing chest pressure")
    assert flag is not None
    assert "108" in urgent_reply(flag)
