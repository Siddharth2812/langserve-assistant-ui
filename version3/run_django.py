import os
import sys
from django.core.management import execute_from_command_line

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "version3.settings")
    # Run Django on port 8001
    sys.argv = ["manage.py", "runserver", "8001"]
    execute_from_command_line(sys.argv) 