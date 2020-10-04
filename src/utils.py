import os
try:
    import git
except:
    pass

def get_root():
    try:
        root = git.Repo(os.getcwd(), search_parent_directories=True).git.rev_parse('--show-toplevel')
    except:
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return root

