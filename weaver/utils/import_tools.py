from importlib.util import spec_from_file_location, module_from_spec


def import_module(path, name='_mod'):
    spec = spec_from_file_location(name, path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
