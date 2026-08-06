import core.auth as auth


def login(token):
    return auth.validate(token)
