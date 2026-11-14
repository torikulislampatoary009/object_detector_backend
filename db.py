"""MySQL access for Aiven.

Credentials are read from environment variables only, so the same code runs
locally (.env file) and on Render (Environment Variables tab):

    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    DB_SSL_MODE  = DISABLED | REQUIRED | VERIFY   (Aiven needs REQUIRED or VERIFY)
    DB_SSL_CA    = full text of Aiven's ca.pem     (only for VERIFY)
"""
import logging
import os
import ssl
import tempfile
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

log = logging.getLogger("db")

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS detections (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        source        VARCHAR(20)  NOT NULL DEFAULT 'upload',
        filename      VARCHAR(255) NULL,
        image_width   INT          NOT NULL,
        image_height  INT          NOT NULL,
        pedestrians   INT          NOT NULL DEFAULT 0,
        cars          INT          NOT NULL DEFAULT 0,
        inference_ms  FLOAT        NOT NULL DEFAULT 0,
        created_at    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    """
    CREATE TABLE IF NOT EXISTS detection_boxes (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        detection_id  INT          NOT NULL,
        label         VARCHAR(30)  NOT NULL,
        confidence    FLOAT        NOT NULL,
        x1            FLOAT        NOT NULL,
        y1            FLOAT        NOT NULL,
        x2            FLOAT        NOT NULL,
        y2            FLOAT        NOT NULL,
        CONSTRAINT fk_detection FOREIGN KEY (detection_id)
            REFERENCES detections(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
]

_schema_ready = False
_ca_file = None


def is_configured():
    return all(os.getenv(k) for k in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAME"))


def _ssl_context():
    mode = os.getenv("DB_SSL_MODE", "REQUIRED").strip().upper()
    if mode == "DISABLED":
        return None

    if mode == "VERIFY":
        global _ca_file
        pem = os.getenv("DB_SSL_CA", "").replace("\\n", "\n").strip()
        if not pem:
            raise RuntimeError("DB_SSL_MODE=VERIFY needs DB_SSL_CA (contents of ca.pem)")
        if _ca_file is None:
            f = tempfile.NamedTemporaryFile("w", suffix=".pem", delete=False)
            f.write(pem + "\n")
            f.close()
            _ca_file = f.name
        ctx = ssl.create_default_context(cafile=_ca_file)
        ctx.check_hostname = False  # Aiven certs are issued to the project CA
        return ctx

    # REQUIRED: encrypted connection without certificate verification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def get_connection():
    if not is_configured():
        raise RuntimeError("Database is not configured: set DB_HOST, DB_USER, DB_PASSWORD, DB_NAME")
    kwargs = dict(
        host=os.environ["DB_HOST"],
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=10,
        read_timeout=20,
        write_timeout=20,
        autocommit=False,
    )
    ctx = _ssl_context()
    if ctx is not None:
        kwargs["ssl"] = ctx
    return pymysql.connect(**kwargs)


@contextmanager
def cursor(commit=False):
    """Open a short-lived connection per request (friendly to free-tier limits)."""
    global _schema_ready
    conn = get_connection()
    try:
        if not _schema_ready:
            _create_tables(conn)
        with conn.cursor() as cur:
            yield cur
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _create_tables(conn):
    global _schema_ready
    with conn.cursor() as cur:
        for stmt in SCHEMA:
            cur.execute(stmt)
    conn.commit()
    _schema_ready = True
    log.info("Database tables ready")


def init_db():
    """Try to create tables at startup. Never crashes the app if the DB is down."""
    if not is_configured():
        log.warning("DB env vars not set - history will be disabled")
        return False
    try:
        conn = get_connection()
        try:
            _create_tables(conn)
        finally:
            conn.close()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Database not reachable at startup: %s", exc)
        return False


def ping():
    try:
        with cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False
