import re
from agent.tools.random_utils import generate_uuid, random_int, random_choice


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def test_uuid_v4():
    out = generate_uuid.invoke({"version": 4})
    assert _UUID_RE.match(out), f"不是合法 UUID: {out}"


def test_uuid_v1():
    out = generate_uuid.invoke({"version": 1})
    assert _UUID_RE.match(out), f"不是合法 UUID: {out}"


def test_random_int_range():
    for _ in range(20):
        out = random_int.invoke({"min_val": 1, "max_val": 10})
        val = int(out)
        assert 1 <= val <= 10


def test_random_int_bad_range():
    out = random_int.invoke({"min_val": 10, "max_val": 1})
    assert "错误" in out


def test_random_choice():
    out = random_choice.invoke({"items": "苹果, 香蕉, 橙子"})
    assert out in ("苹果", "香蕉", "橙子")
