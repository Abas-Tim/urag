from core.http import fetch as http_fetch


def get_user():
    return http_fetch("/user")


def get_item():
    return http_fetch("/item")
