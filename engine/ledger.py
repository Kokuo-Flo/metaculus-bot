"""SQLite ledger: every euro in or out, every operation, every heartbeat.

Money rows carry a `source`:
- 'measured'  → a real transaction or a provider-reported cost (counts toward proof levels)
- 'estimated' → a model output (paper trading); shown separately, never mixed into ROI
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("REVENUE_LAB_LEDGER") or Path(__file__).resolve().parent.parent / "data" / "ledger.sqlite")

COST_KINDS = ("api_cost", "electricity", "infrastructure", "platform_fee", "transaction_fee", "depreciation")
SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy (
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('candidate','paper','active','abandoned')),
    proof_level INTEGER NOT NULL DEFAULT 0, capital_eur REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, status_changed_at TEXT NOT NULL, status_reason TEXT);
CREATE TABLE IF NOT EXISTS money (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT NOT NULL REFERENCES strategy(id),
    ts TEXT NOT NULL, kind TEXT NOT NULL CHECK (kind IN ('revenue','capital',
        'api_cost','electricity','infrastructure','platform_fee','transaction_fee','depreciation')),
    amount_eur REAL NOT NULL, source TEXT NOT NULL CHECK (source IN ('measured','estimated')), note TEXT);
CREATE TABLE IF NOT EXISTS operation (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT NOT NULL REFERENCES strategy(id),
    ts TEXT NOT NULL, kind TEXT NOT NULL, ok INTEGER NOT NULL, duration_s REAL, detail TEXT);
CREATE TABLE IF NOT EXISTS heartbeat (ts TEXT NOT NULL, strategy_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evaluation (
    id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id TEXT NOT NULL, ts TEXT NOT NULL,
    expected_net_eur_month REAL, verdict TEXT NOT NULL, detail TEXT);
CREATE INDEX IF NOT EXISTS money_strategy_ts ON money(strategy_id, ts);
CREATE INDEX IF NOT EXISTS operation_strategy_ts ON operation(strategy_id, ts);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path | None = None):
        path = path or DEFAULT_DB
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    # --- Strategies ---
    def register(self, strategy_id: str, name: str, status: str = "candidate", proof_level: int = 0) -> None:
        self.connection.execute(
            "INSERT OR IGNORE INTO strategy (id, name, status, proof_level, created_at, status_changed_at)"
            " VALUES (?,?,?,?,?,?)", (strategy_id, name, status, proof_level, now(), now()))
        self.connection.commit()

    def set_status(self, strategy_id: str, status: str, reason: str, proof_level: int | None = None) -> None:
        self.connection.execute(
            "UPDATE strategy SET status=?, status_reason=?, status_changed_at=?,"
            " proof_level=COALESCE(?, proof_level) WHERE id=?",
            (status, reason, now(), proof_level, strategy_id))
        self.connection.commit()

    def strategies(self) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM strategy ORDER BY created_at").fetchall()

    # --- Money / operations ---
    def money(self, strategy_id: str, kind: str, amount_eur: float, source: str = "measured", note: str = "") -> None:
        self.connection.execute(
            "INSERT INTO money (strategy_id, ts, kind, amount_eur, source, note) VALUES (?,?,?,?,?,?)",
            (strategy_id, now(), kind, amount_eur, source, note))
        if kind == "capital" and source == "measured":
            self.connection.execute("UPDATE strategy SET capital_eur = capital_eur + ? WHERE id=?",
                                    (amount_eur, strategy_id))
        self.connection.commit()

    def operation(self, strategy_id: str, kind: str, ok: bool, duration_s: float = 0.0, **detail) -> None:
        self.connection.execute(
            "INSERT INTO operation (strategy_id, ts, kind, ok, duration_s, detail) VALUES (?,?,?,?,?,?)",
            (strategy_id, now(), kind, int(ok), duration_s, json.dumps(detail, default=str)))
        self.connection.commit()

    def heartbeat(self, strategy_id: str) -> None:
        self.connection.execute("INSERT INTO heartbeat VALUES (?,?)", (now(), strategy_id))
        self.connection.commit()

    def evaluation(self, strategy_id: str, expected_net_eur_month: float | None, verdict: str, **detail) -> None:
        self.connection.execute(
            "INSERT INTO evaluation (strategy_id, ts, expected_net_eur_month, verdict, detail) VALUES (?,?,?,?,?)",
            (strategy_id, now(), expected_net_eur_month, verdict, json.dumps(detail, default=str)))
        self.connection.commit()

    def latest_evaluation(self, strategy_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM evaluation WHERE strategy_id=? ORDER BY id DESC LIMIT 1", (strategy_id,)).fetchone()

    # --- Economics ---
    def economics(self, strategy_id: str | None = None, days: int | None = None, source: str = "measured") -> dict:
        """net_profit = revenue - electricity - API - infra - platform fees - transaction fees - depreciation."""
        clauses, params = ["source = ?"], [source]
        if strategy_id:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        if days:
            clauses.append("ts >= ?")
            params.append((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        where = " AND ".join(clauses)
        totals = {row["kind"]: row["total"] for row in self.connection.execute(
            f"SELECT kind, SUM(amount_eur) AS total FROM money WHERE {where} GROUP BY kind", params)}
        first = self.connection.execute(f"SELECT MIN(ts) AS first FROM money WHERE {where}", params).fetchone()["first"]
        revenue = totals.get("revenue", 0.0)
        costs = {kind: totals.get(kind, 0.0) for kind in COST_KINDS}
        net = revenue - sum(costs.values())
        capital = totals.get("capital", 0.0)
        hours = max(1.0, (datetime.now(timezone.utc) - datetime.fromisoformat(first)).total_seconds() / 3600) if first else 0.0
        per_hour = net / hours if hours else 0.0
        per_month = per_hour * 24 * 30.44
        return {
            "revenue": revenue, "costs": costs, "total_costs": sum(costs.values()), "net_profit": net,
            "capital": capital, "hours": hours, "eur_per_hour": per_hour, "eur_per_day": per_hour * 24,
            "eur_per_month": per_month,
            "roi": (net / capital) if capital else None,
            "payback_months": (capital / per_month) if capital and per_month > 0 else None,
        }

    def operations_stats(self, strategy_id: str | None = None) -> dict:
        where, params = ("WHERE strategy_id=?", [strategy_id]) if strategy_id else ("", [])
        row = self.connection.execute(
            f"SELECT COUNT(*) AS total, SUM(ok) AS ok FROM operation {where}", params).fetchone()
        total = row["total"] or 0
        return {"operations": total, "success_rate": (row["ok"] or 0) / total if total else None}

    def uptime(self, strategy_id: str, expected_interval_s: int, days: int = 7) -> float | None:
        since = datetime.now(timezone.utc) - timedelta(days=days)
        rows = self.connection.execute(
            "SELECT ts FROM heartbeat WHERE strategy_id=? AND ts>=? ORDER BY ts",
            (strategy_id, since.isoformat())).fetchall()
        if not rows:
            return None
        first = datetime.fromisoformat(rows[0]["ts"])
        expected = max(1, (datetime.now(timezone.utc) - first).total_seconds() / expected_interval_s)
        return min(1.0, len(rows) / expected)
