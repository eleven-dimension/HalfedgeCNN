import importlib.util
import inspect

def register_neighbor_functions(module_name):
    neighbor_func_map = {}
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and name.startswith('get_neighbors_'):
                    neighbor_type = name.split('_')[-1]
                    neighbor_func_map[neighbor_type] = obj
    except ImportError as e:
        print(f"Failed to import module {module_name}: {e}")
    return neighbor_func_map
