import os

# Statement for enabling the development environment
DEBUG = os.environ.get('DEBUG', False)

# Define the application directory
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, '/web/static')

# Application threads. A utils general assumption is
# using 2 per available processor cores - to handle
# incoming requests using one and performing background
# operations using the other.
THREADS_PER_PAGE = 2

# Enable protection agains *Cross-site Request Forgery (CSRF)*
CSRF_ENABLED = True

# Use a secure, unique and absolutely secret key for
# signing the data.
CSRF_SESSION_KEY = "secret"

# Secret key for signing cookies
SECRET_KEY = os.environ.get('JWT_ENCRYPTION_KEY', 'secret')
