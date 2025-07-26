


class SummaryNode:
    def __init__(self, summary_text, children=None, qa_pairs=None):
        self.summary = summary_text          # The summary at this level
        self.qa_pairs = qa_pairs or []       # List of (Q, A) tuples
        self.children = children or []       # Child nodes (segments/subsummaries)

    def add_child(self, child_node):
        self.children.append(child_node)

