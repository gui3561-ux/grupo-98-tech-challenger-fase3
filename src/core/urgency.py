from enum import StrEnum


class UrgencyLevel(StrEnum):
    """Categorias de urgência atribuídas a um laudo médico."""

    NORMAL = "normal"
    ATENCAO = "atencao"
    URGENTE = "urgente"


URGENCY_ORDER: dict[UrgencyLevel, int] = {
    UrgencyLevel.NORMAL: 0,
    UrgencyLevel.ATENCAO: 1,
    UrgencyLevel.URGENTE: 2,
}

# Palavras-chave que indicam gravidade crescente nos laudos sintéticos.
KEYWORDS_NORMAL: tuple[str, ...] = ("sem achados", "normal", "sem alterações", "exame normal")
KEYWORDS_ATENCAO: tuple[str, ...] = ("recomenda", "leve", "acompanhamento", "moderada", "observar")
KEYWORDS_URGENTE: tuple[str, ...] = ("emergência", "grave", "urgente", "sangramento", "crítico", "internação")
