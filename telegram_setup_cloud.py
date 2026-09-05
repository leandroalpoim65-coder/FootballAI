from __future__ import annotations

import os

from notify_signals_cloud import (
    discover_telegram_chats,
    send_telegram,
)


def main() -> None:
    print()
    print("==========================================")
    print(" FOOTBALLAI — TELEGRAM SETUP / TEST")
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
            "❌ TELEGRAM_BOT_TOKEN não configurado."
        )
        return

    if not chat_id:
        print(
            "TELEGRAM_CHAT_ID ainda não configurado."
        )
        print(
            "Envia /start ao bot no Telegram. "
            "Vou procurar os chats disponíveis."
        )
        print()

        chats = discover_telegram_chats(
            bot_token
        )

        if not chats:
            print(
                "⚠ Nenhum chat encontrado."
            )
            print(
                "Abre o bot no Telegram, envia /start "
                "e volta a executar este workflow."
            )
            return

        print(
            "Chats Telegram detetados:"
        )

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
            "Copia o teu ID para o GitHub secret "
            "TELEGRAM_CHAT_ID."
        )
        return

    message_id = send_telegram(
        bot_token,
        chat_id,
        (
            "✅ FootballAI — Telegram configurado\n\n"
            "As notificações dos sinais de PAPER TRADE "
            "estão prontas.\n"
            "O threshold real continua congelado em 55%."
        ),
    )

    print(
        "✅ Mensagem de teste enviada com sucesso."
    )
    print(
        f"Telegram message_id: {message_id or '—'}"
    )


if __name__ == "__main__":
    main()
