from backend.models.data_schemas import ThreatEvent

_threats: list[ThreatEvent] = []


def add_threat(event: ThreatEvent) -> None:
    _threats.append(event)


def get_threats() -> list[ThreatEvent]:
    return list(_threats)


def get_latest_threat() -> ThreatEvent | None:
    if not _threats:
        return None
    return _threats[-1]
