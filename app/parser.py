import ast


def parse_python_file(source: str):
    """
    Parse Python source code and extract classes and functions.
    """

    tree = ast.parse(source)

    results = {
        "classes": [],
        "functions": [],
        "imports": []
    }

    # Extract classes and functions
    for node in tree.body:

        # Class
        if isinstance(node, ast.ClassDef):

            results["classes"].append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "source": ast.get_source_segment(source, node)
            })

            # Methods inside the class
            for child in node.body:

                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    results["functions"].append({
                        "name": f"{node.name}.{child.name}",
                        "start_line": child.lineno,
                        "end_line": child.end_lineno,
                        "source": ast.get_source_segment(source, child)
                    })

        # Standalone functions
        elif isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef)
        ):

            results["functions"].append({
                "name": node.name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "source": ast.get_source_segment(source, node)
            })

    # Extract imports
    for node in ast.walk(tree):

        if isinstance(node, ast.Import):

            for alias in node.names:
                results["imports"].append(alias.name)

        elif isinstance(node, ast.ImportFrom):

            if node.module:
                results["imports"].append(node.module)

    return results