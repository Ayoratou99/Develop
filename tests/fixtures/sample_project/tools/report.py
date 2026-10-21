"""Petit script d'export de rapport (exemple Python pour les tests d'analyse)."""
import csv
import json
from dataclasses import dataclass
from typing import List


@dataclass
class OrderRow:
    reference: str
    status: str
    total_amount: int


def load_orders(path: str) -> List[OrderRow]:
    with open(path, encoding="utf-8") as fh:
        return [OrderRow(**row) for row in json.load(fh)]


def export_csv(rows: List[OrderRow], out: str) -> None:
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["reference", "status", "total_amount"])
        for r in rows:
            writer.writerow([r.reference, r.status, r.total_amount])


class ReportBuilder:
    def __init__(self, rows: List[OrderRow]):
        self.rows = rows

    def total(self) -> int:
        return sum(r.total_amount for r in self.rows)
