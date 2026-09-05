from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import libsql


APP_TIMEZONE = ZoneInfo("Europe/Lisbon")
PAPER_TABLE = "paper_trades"
TELEGRAM_API_BASE = "https://api.telegram.org"


def connect_turso():
    url = os.getenv(
        "TURSO_DATABASE_URL",
        "",
    ).strip()

    token = os.getenv(
        "TURSO_AUTH_TOKEN",
        "",
    ).strip()

    if not url:
        raise RuntimeError(
            "TURSO_DATABASE_URL não configurada."
        )

    if not token:
        raise RuntimeError(
            "TURSO_AUTH_TOKEN não configurado."
        )

    return libsql.connect(
        database=url,
        auth_token=token,
    )


def table_exists(
    connection,
    table_name: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def table_columns(
    connection,
    table_name: str,
) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
    }


def ensure_notification_columns(
    connection,
) -> None:
    columns = table_columns(
        connection,
        PAPER_TABLE,
    )

    if "telegram_notified_at" not in columns:
        connection.execute(
            f"""
            ALTER TABLE {PAPER_TABLE}
            ADD COLUMN telegram_notified_at TEXT
            """
        )

    if "telegram_message_id" not in columns:
        connection.execute(
            f"""
            ALTER TABLE {PAPER_TABLE}
            ADD COLUMN telegram_message_id TEXT
            """
        )

    connection.commit()


def fetch_dicts(
    connection,
    sql: str,
    params: tuple = (),
) -> list[dict]:
    cursor = connection.execute(
        sql,
        params,
    )
    rows = cursor.fetchall()
    columns = [
        item[0]
        for item in (cursor.description or [])
    ]

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def safe_float(
    value: object,
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if number != number:
        return None

    return number


def parse_fixture_datetime(
    fixture_date: object,
    fixture_time: object,
) -> datetime | None:
    date_text = str(
        fixture_date or ""
    ).strip()

    if not date_text:
        return None

    time_text = str(
        fixture_time or "00:00"
    ).strip()

    if not time_text:
        time_text = "00:00"

    candidates = [
        f"{date_text[:10]} {time_text[:8]}",
        f"{date_text[:10]} {time_text[:5]}",
    ]

    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(
                candidate
            )
        except ValueError:
            continue

        return parsed.replace(
            tzinfo=APP_TIMEZONE
        )

    return None


def format_number(
    value: object,
    decimals: int = 2,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "—"

    return f"{number:.{decimals}f}"


def format_percent(
    value: object,
    decimals: int = 1,
) -> str:
    number = safe_float(
        value
    )

    if number is None:
        return "—"

    return f"{number * 100:.{decimals}f}%"


def build_message(
    row: dict,
) -> str:
    fixture_date = str(
        row.get("fixture_date") or "—"
    )
    fixture_time = str(
        row.get("fixture_time") or "—"
    )
    league = str(
        row.get("league") or "—"
    )
    home_team = str(
        row.get("home_team") or "—"
    )
    away_team = str(
        row.get("away_team") or "—"
    )

    odd = format_number(
        row.get("odd_over25_open"),
        3,
    )
    probability = format_percent(
        row.get("clv_probability"),
        1,
    )
    threshold = format_percent(
        row.get("signal_threshold"),
        1,
    )
    stake = format_number(
        row.get("planned_stake"),
        2,
    )

    return (
        "🎯 FootballAI — NOVO SINAL\n"
        "🧪 PAPER TRADE — não é aposta real\n\n"
        f"⚽ {home_team} vs {away_team}\n"
        f"🏆 {league}\n"
        f"📅 {fixture_date} · {fixture_time}\n\n"
        "Mercado: Over 2.5\n"
        f"Odd OPEN: {odd}\n"
        f"Prob. CLV+: {probability}\n"
        f"Threshold atual: {threshold}\n"
        f"Stake paper: {stake}u\n\n"
        "ℹ️ clv_probability estima a probabilidade "
        "de CLV positivo; não é a probabilidade "
        "de o jogo terminar Over 2.5."
    )


def discover_telegram_chats(
    bot_token: str,
) -> list[dict]:
    url = (
        f"{TELEGRAM_API_BASE}/bot"
        f"{bot_token}/getUpdates"
    )

    request = Request(
        url,
        headers={
            "User-Agent":
                "FootballAI-Telegram-Notifier/1.0",
            "Accept":
                "application/json",
        },
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:
            raw = response.read()
    except Exception as error:
        raise RuntimeError(
            f"Não foi possível ler getUpdates: {error}"
        ) from error

    try:
        result = json.loads(
            raw.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            "Resposta inválida do Telegram."
        ) from error

    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(
            "Telegram recusou getUpdates."
        )

    updates = result.get("result", [])
    if not isinstance(updates, list):
        return []

    chats: dict[str, dict] = {}

    for update in updates:
        if not isinstance(update, dict):
            continue

        message = (
            update.get("message")
            or update.get("edited_message")
            or update.get("channel_post")
        )

        if not isinstance(message, dict):
            continue

        chat = message.get("chat")
        if not isinstance(chat, dict):
            continue

        chat_id = chat.get("id")
        if chat_id is None:
            continue

        key = str(chat_id)
        chats[key] = {
            "id": key,
            "type": str(
                chat.get("type") or ""
            ),
            "username": str(
                chat.get("username") or ""
            ),
            "first_name": str(
                chat.get("first_name") or ""
            ),
            "title": str(
                chat.get("title") or ""
            ),
        }

    return list(
        chats.values()
    )


def send_telegram(
    bot_token: str,
    chat_id: str,
    text: str,
) -> str:
    url = (
        f"{TELEGRAM_API_BASE}/bot"
        f"{bot_token}/sendMessage"
    )

    payload = urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
    ).encode(
        "utf-8"
    )

    request = Request(
        url,
        data=payload,
        headers={
            "Content-Type":
                "application/x-www-form-urlencoded",
            "User-Agent":
                "FootballAI-Telegram-Notifier/1.0",
        },
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=30,
        ) as response:
            raw = response.read()
    except HTTPError as error:
        try:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            detail = ""

        raise RuntimeError(
            f"Telegram HTTP {error.code}: "
            f"{detail[:400]}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Erro de ligação ao Telegram: "
            f"{error.reason}"
        ) from error

    try:
        result = json.loads(
            raw.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(
            "Resposta inválida do Telegram."
        ) from error

    if not isinstance(result, dict):
        raise RuntimeError(
            "Resposta inesperada do Telegram."
        )

    if not result.get("ok"):
        raise RuntimeError(
            "Telegram recusou a mensagem: "
            f"{result.get('description')}"
        )

    message = result.get("result")
    if not isinstance(message, dict):
        return ""

    return str(
        message.get("message_id") or ""
    )


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — TELEGRAM SIGNAL NOTIFIER")
    print("==========================================")
    print()

    bot_token = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    ).strip()

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID",
        "",
    ).strip()

    if not bot_token:
        print(
            "⚪ TELEGRAM_BOT_TOKEN ainda não configurado."
        )
        return

    if not chat_id:
        print(
            "⚪ TELEGRAM_CHAT_ID ainda não configurado."
        )
        print(
            "Envia /start ao bot no Telegram e volta a executar "
            "este workflow. Os chats detetados aparecem abaixo."
        )

        try:
            chats = discover_telegram_chats(
                bot_token
            )
        except Exception as error:
            print(
                f"⚠ Não foi possível descobrir chats: {error}"
            )
            return

        if not chats:
            print(
                "Nenhum chat encontrado. Envia primeiro /start ao bot."
            )
            return

        print()
        print("Chats Telegram detetados:")

        for chat in chats:
            label = (
                chat.get("username")
                or chat.get("first_name")
                or chat.get("title")
                or "sem nome"
            )

            print(
                f"  - ID {chat['id']} | "
                f"{chat.get('type') or '?'} | "
                f"{label}"
            )

        print()
        print(
            "Adicionar o ID correto ao secret TELEGRAM_CHAT_ID."
        )
        return

    connection = connect_turso()

    try:
        if not table_exists(
            connection,
            PAPER_TABLE,
        ):
            print(
                "⚪ A tabela paper_trades ainda não existe."
            )
            return

        ensure_notification_columns(
            connection
        )

        today = datetime.now(
            APP_TIMEZONE
        ).date().isoformat()

        rows = fetch_dicts(
            connection,
            f"""
            SELECT *
            FROM {PAPER_TABLE}
            WHERE signal = 1
              AND COALESCE(status, 'OPEN') = 'OPEN'
              AND telegram_notified_at IS NULL
              AND date(fixture_date) >= date(?)
            ORDER BY
                fixture_date,
                fixture_time,
                id
            """,
            (today,),
        )

        now = datetime.now(
            APP_TIMEZONE
        )

        pending: list[dict] = []
        skipped_past = 0

        for row in rows:
            fixture_datetime = parse_fixture_datetime(
                row.get("fixture_date"),
                row.get("fixture_time"),
            )

            if (
                fixture_datetime is not None
                and fixture_datetime <= now
            ):
                skipped_past += 1
                continue

            pending.append(
                row
            )

        print(
            f"Sinais por notificar       : {len(pending)}"
        )

        if skipped_past:
            print(
                f"Sinais já iniciados        : {skipped_past}"
            )

        if not pending:
            print(
                "✅ Nenhum novo sinal para enviar."
            )
            return

        sent = 0
        failed = 0

        for row in pending:
            row_id = int(
                row["id"]
            )
            message = build_message(
                row
            )

            try:
                message_id = send_telegram(
                    bot_token,
                    chat_id,
                    message,
                )
            except Exception as error:
                failed += 1
                print(
                    f"❌ {row.get('home_team')} vs "
                    f"{row.get('away_team')} | {error}"
                )
                continue

            notified_at = (
                datetime.now(
                    timezone.utc
                )
                .replace(
                    microsecond=0
                )
                .isoformat()
            )

            connection.execute(
                f"""
                UPDATE {PAPER_TABLE}
                SET
                    telegram_notified_at = ?,
                    telegram_message_id = ?
                WHERE id = ?
                  AND telegram_notified_at IS NULL
                """,
                (
                    notified_at,
                    message_id or None,
                    row_id,
                ),
            )

            connection.commit()

            sent += 1

            print(
                f"✅ Telegram enviado | "
                f"{row.get('fixture_date')} "
                f"{row.get('fixture_time')} | "
                f"{row.get('home_team')} vs "
                f"{row.get('away_team')}"
            )

        print()
        print("==========================================")
        print(" RESUMO TELEGRAM")
        print("==========================================")
        print(
            f"Enviados                  : {sent}"
        )
        print(
            f"Falhados                  : {failed}"
        )
        print()
        print(
            "Cada sinal é marcado na Turso depois "
            "de o Telegram confirmar o envio."
        )

    finally:
        connection.close()


if __name__ == "__main__":
    main()
