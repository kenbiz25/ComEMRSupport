import os
import sys
# ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.flow import BotFlow

bf = BotFlow(None, None, None)

# test 1
text = "Hello! *Please* follow these steps:\n1. Do this\n2. Do that"
out = bf._format_outgoing(text)
assert "*" not in out
assert "1. Do this" in out

# test 2
text = "Steps:\n1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n7. g"
out = bf._format_outgoing(text)
assert "6." not in out
assert "..." in out

# test 3
text = "1. One\n2. Two\n3. Three\n4. Four\n5."  # last item incomplete
out = bf._format_outgoing(text)
assert "5." not in out or out.strip().endswith('...') or out.strip().endswith('.')

print('formatting checks ok')
