# Engine API

Base: FastAPI app in `engine/api/server.py` (optional: `pip install -e '.[api]'`).

## Endpoints (contract)

| Method | Path | Status |
|--------|------|--------|
| GET | `/health` | Phase 1 |
| POST | `/projects/load` | Phase 6 |
| POST | `/models/load` | Phase 6 |
| GET | `/models/{id}/graph` | Phase 6 |
| POST | `/profile` | Phase 6 |
| GET | `/profile/{id}` | Phase 6 |
| POST | `/cache/analyze` | Phase 6 |
| POST | `/cache/experiment` | Phase 6 |
| POST | `/optimize` | Phase 6 |
| POST | `/benchmark` | Phase 6 |
| GET | `/experiments` | Phase 6 |
| GET | `/experiments/{id}` | Phase 6 |
| POST | `/agent/chat` | Phase 6 |
| POST | `/patch/apply` | Phase 6 |
| POST | `/patch/revert` | Phase 6 |

The UI must call these endpoints only — no optimization logic in the frontend.
