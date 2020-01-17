class CacheContext:

    def __init__(self):
        self.context_tree = {}
        self.current_node = self.context_tree
        self.parent_stack = [self.current_node]

    def __enter__(self):
        cache_context = self
        cache = self.current_node
        return cache_context

    def __exit__(self, type, value, traceback):
        cache_context = None
        cache = {}


cache_context: CacheContext = None
cache = {}


class SubContext:

    def __init__(self, name):
        self.name = name
        if cache_context is not None:
            if name in cache_context.current_node:
                self.node = cache_context.current_node[name]
            else:
                self.node = {}
                cache_context.current_node[name] = self.node

    def __enter__(self):
        if cache_context is not None:
            cache_context.parent_stack.append(cache_context.current_node)
            cache_context.current_node = cache_context.current_node[self.name]
            cache = cache_context.current_node
            return self.node
        return {}

    def __exit__(self, exc_type, exc_val, exc_tb):
        if cache_context is not None:
            cache_context.current_node = cache_context.parent_stack.pop()
            cache = cache_context.current_node
