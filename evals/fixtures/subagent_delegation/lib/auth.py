"""User authentication helpers: password hashing and session-token
issuance. Kept narrow — no role-based access control yet."""
import hashlib


def hash_password(plain: str) -> str:
    return hashlib.sha256(plain.encode()).hexdigest()


def issue_token(user_id: int) -> str:
    return f"session-{user_id}-token"
