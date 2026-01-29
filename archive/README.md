Archived apps and utilities

This directory holds archived applications and utilities that are not part of the main production codebase but are retained for future reference.

- `apps/whatsapp_gateway` — moved here on 2026-01-29. Contains a simple webhook handler and media upload helper that were superseded by the main application logic. If you need to restore it, copy files back to `apps/whatsapp_gateway/` and remove the stub files in that directory.

Rationale: Archiving keeps the repository clean for production while preserving useful code for future reference or re-use.
