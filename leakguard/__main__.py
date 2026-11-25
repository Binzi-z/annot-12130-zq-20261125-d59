"""支持 python -m leakguard <command> 形式调用。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
