import time

import pytest
from fastapi import HTTPException
from unittest.mock import patch

from agent.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    get_current_user,
)


class FakeSettings:
    jwt_secret = "test-jwt-secret-with-at-least-32-characters!!"
    token_expire_hours = 168


def test_hash_and_verify_password():
    """bcrypt 哈希后验证正确密码。"""
    hashed = hash_password("mypassword")
    assert hashed != "mypassword"
    assert verify_password("mypassword", hashed)


def test_verify_wrong_password():
    """错误密码验证失败。"""
    hashed = hash_password("mypassword")
    assert not verify_password("wrong", hashed)


def test_create_and_decode_token():
    """创建 JWT 令牌后解码，payload 字段一致。"""
    with patch("agent.auth.Settings", return_value=FakeSettings()):
        token = create_access_token(1, "alice")
        payload = decode_token(token)
    assert payload["sub"] == "1"
    assert payload["username"] == "alice"


def test_decode_invalid_token():
    """无效令牌解码时抛异常。"""
    with patch("agent.auth.Settings", return_value=FakeSettings()):
        with pytest.raises(Exception):
            decode_token("invalid.token.here")


def test_get_current_user_valid():
    """有效令牌解析出用户信息。"""
    with patch("agent.auth.Settings", return_value=FakeSettings()):
        token = create_access_token(42, "bob")
        user = get_current_user(token=token)
    assert user == {"id": 42, "username": "bob"}


def test_get_current_user_invalid():
    """无效令牌触发 401 HTTPException。"""
    with patch("agent.auth.Settings", return_value=FakeSettings()):
        with pytest.raises(HTTPException) as exc_info:
            get_current_user(token="invalid")
    assert exc_info.value.status_code == 401
