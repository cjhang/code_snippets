import sys
import os

# append the user's python library
UserHome = os.path.expanduser('~')
sys.path.append(os.path.join(UserHome, 'work/projects/code_snippets/casa'))

# append the analysisUtils
sys.path.append(os.path.join(UserHome,"applications/casa/analysis_scripts/"))

try:
    import analysisUtils as au
except:
    print("No analysisUtils is found. Skip...")
