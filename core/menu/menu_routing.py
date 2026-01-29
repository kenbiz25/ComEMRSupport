def show_menu_options(context):
    """
    Triggered when confidence is low
    context: dict of user input, role, kb results
    """
    menu = [
        "1. Kindly repeat your question",
        "2. Please choose from common queries below",
        "3. Speak to Support Agent",
    ]
    return menu
