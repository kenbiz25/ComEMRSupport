"""
core/tickets/ticket_manager.py

Writes support tickets to an Excel file (tickets.xlsx) using openpyxl.
Each ticket row: Ticket ID | Phone Number | Issue | Status | Created At
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    openpyxl = None  # type: ignore
    _OPENPYXL_AVAILABLE = False

_HEADERS = ["Ticket ID", "Phone Number", "Issue", "Status", "Created At"]


def _ticket_path() -> Path:
    from config.settings import settings
    return Path(getattr(settings, "TICKET_FILE_PATH", "tickets.xlsx"))


def _init_workbook(path: Path) -> None:
    """Create the Excel file with headers if it doesn't exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Tickets"
    ws.append(_HEADERS)
    # Basic column widths for readability
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 60
    ws.column_dimensions["D"].width = 10
    ws.column_dimensions["E"].width = 20
    wb.save(str(path))


def create_ticket(phone_number: str, issue: str, status: str = "Open") -> Optional[str]:
    """
    Append a new ticket row to the Excel file and return the ticket ID.
    Returns None if openpyxl is not installed or write fails.
    """
    if not _OPENPYXL_AVAILABLE:
        logger.error("openpyxl is not installed — cannot create ticket. Run: pip install openpyxl")
        return None

    try:
        path = _ticket_path()
        _init_workbook(path)

        ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        wb = openpyxl.load_workbook(str(path))
        ws = wb["Tickets"]
        ws.append([ticket_id, phone_number, issue[:500], status, created_at])
        wb.save(str(path))

        logger.info("Ticket created | id=%s phone=%s", ticket_id, phone_number)
        return ticket_id

    except Exception:
        logger.exception("Failed to create ticket for %s", phone_number)
        return None
