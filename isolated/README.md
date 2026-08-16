# Isolated — Not Part of Backend 2 Simulator

This notice documents code present in the repository that is **not** part of the SIH acoustic cybersecurity Backend 2 simulator pipeline.

## `Reverse_Shell-main/`

| Item | Detail |
|------|--------|
| **Contents** | `reverse.py`, `shell.py`, `keylogger.py` — TCP reverse shell, screenshot capture (`mss`), keyboard logging (`pynput`) |
| **Relation to SIH** | Educational/offensive reference material; **not** used by `backend/` encode/modulate/WAV pipeline |
| **Action** | **Isolated** — kept for team reference; not integrated into simulator or threat API |
| **Do not** | Run against unauthorized systems; not required for `samples/backend2/` handoff |

The Backend 2 simulator produces controlled BFSK WAV files via `backend/services/payload_service.py`. It does not depend on this folder.

## Unused schema stubs in `backend/models/data_schemas.py`

`ReverseShellCommandRequest` and `ReverseShellStatus` are legacy stubs with **no registered API routes**. They are not part of the integration contract.

## Removed from main `requirements.txt`

`mss` and `pynput` were removed from project dependencies because only `Reverse_Shell-main/` used them. Install separately if studying that isolated folder:

```bash
pip install mss pynput
```
