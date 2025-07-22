from functools import wraps
from flask import request, abort
import os

def require_api_key():
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = request.headers.get('x-api-key')
            expected = os.getenv("PYTHON_API_KEY")
            if not api_key or api_key != expected:
                abort(403)
            return f(*args, **kwargs)
        return decorated_function
    return decorator
