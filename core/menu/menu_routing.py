def show_menu_options(context=None):
    """
    Return a short, numbered menu to present to users on first interaction or low-confidence.
    """
    menu = [
        "Welcome to ComEMR Support! Please choose an option below:",
        "1. Report a System Issue\n   App not working, login errors, sync issues",
        "2. Get Help Using ComEMR\n   How to register patients, submit reports, use features",
        "3. Check System Status\n   Downtime, known issues, maintenance updates",
        "4. Request a Feature or Improvement\n   Suggest changes or new functionality",
        "5. Training & User Guides\n   Manuals, videos, onboarding support",
    ]
    return menu


def process_menu_selection(selection: str):
    """Interpret a simple numeric or textual selection and return (reply_text, meta).

    meta may contain keys that help route the request.
    """
    sel = (selection or "").strip().lower()

    # 1: Report a System Issue
    if sel in ("1", "one", "report", "system", "issue", "report a system issue"):
        return (
            "Sorry you're seeing issues. Please briefly describe the problem (error messages, steps to reproduce) and include screenshots if possible. We'll log this and follow up.",
            {"menu_selected": "1"},
        )

    # 2: Get Help Using ComEMR
    if sel in ("2", "two", "help", "using", "get help", "get help using comemr"):
        common = (
            "I can help with using ComEMR. Common tasks:\n1. Register patients\n2. Submit reports\n3. Use features\n\n"
            "Reply with the number (e.g., '1') or describe your issue and I'll assist."
        )
        return (common, {"menu_selected": "2"})

    # 3: Check System Status
    if sel in ("3", "three", "status", "system status", "check system status"):
        try:
            from config.settings import settings
            url = getattr(settings, "SUPPORT_DOCS_URL", "")
        except Exception:
            url = ""
        note = f"Check {url} for status and known issues." if url else "I can check current known issues for you. Would you like me to list recent incidents?"
        return (f"System status: {note}", {"menu_selected": "3"})

    # 4: Feature request
    if sel in ("4", "four", "feature", "request", "feature request", "request a feature"):
        return (
            "Thanks — please describe the feature or improvement you'd like, including the problem it solves and who benefits. We'll forward this to our product team.",
            {"menu_selected": "4"},
        )

    # 5: Training & Guides
    if sel in ("5", "five", "training", "guides", "user guides", "training & user guides"):
        try:
            from config.settings import settings
            url = getattr(settings, "SUPPORT_DOCS_URL", "")
        except Exception:
            url = ""
        note = f"See our docs: {url}" if url else "I can point you to training materials. What topic do you need help with?"
        return (f"Training & user guides: {note}", {"menu_selected": "5"})

    # fallback: ask user to clarify
    return ("I didn't understand that selection. Please reply with a number between 1 and 5.", {"menu_selected": None})
