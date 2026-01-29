import traceback, sys, os
print('CWD:', os.getcwd())
# Ensure project root is on sys.path so we can import packages from repo
repo_root = os.path.abspath(os.path.join(os.getcwd()))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
print('sys.path[0]:', sys.path[0])
try:
    import rag.flow
    print('Imported rag.flow successfully')
    print('BotFlow class:', hasattr(rag.flow, 'BotFlow'))
except Exception as e:
    traceback.print_exc()
    print('\nException type:', type(e))
