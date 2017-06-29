

def remove_latex_chars_from_arrow(direction):
    """Remove non-filename friendly characters from LaTeX direction"""
    return direction.strip('$\\')
