import os
import sys

try:
    import cv2
except Exception as e:
    print(f'[RuntimeHook] cv2 import failed: {e}')

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS

    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    os.environ['PADDLEX_HOME'] = os.path.join(base, '.paddlex')

    paddle_dir = os.path.join(base, 'paddle')
    if os.path.isdir(paddle_dir):
        os.add_dll_directory(paddle_dir)
    os.add_dll_directory(base)
    os.environ['PATH'] = base + os.pathsep + paddle_dir + os.pathsep + os.environ.get('PATH', '')

    def _patch_paddlex_deps():
        try:
            import cv2
            import sys
            import paddlex.utils.deps as deps_mod

            # Inject cv2 into ALL already-loaded paddlex modules
            for mod_name, mod in list(sys.modules.items()):
                if mod_name.startswith('paddlex') and mod is not None:
                    if not hasattr(mod, 'cv2'):
                        try:
                            setattr(mod, 'cv2', cv2)
                        except Exception:
                            pass

            # Hook future paddlex module imports to also get cv2
            import importlib
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def patched_import(name, *args, **kwargs):
                module = original_import(name, *args, **kwargs)
                if name.startswith('paddlex') or (args and len(args) > 1 and isinstance(args[1], dict) and args[1].get('__name__', '').startswith('paddlex')):
                    if not hasattr(module, 'cv2'):
                        try:
                            setattr(module, 'cv2', cv2)
                        except Exception:
                            pass
                return module

            import builtins
            builtins.__import__ = patched_import

            # Patch deps
            deps_mod.is_extra_available.cache_clear()
            deps_mod.is_extra_available = lambda extra: True
            deps_mod.is_dep_available.cache_clear()
            deps_mod.is_dep_available = lambda dep, /, check_version=False: True
            def require_deps(*deps, obj_name=None): pass
            deps_mod.require_deps = require_deps
            def require_extra(extra, *, obj_name=None, alt=None): pass
            deps_mod.require_extra = require_extra
            def pipeline_requires_extra(extra, *, alt=None):
                def _deco(cls): return cls
                return _deco
            deps_mod.pipeline_requires_extra = pipeline_requires_extra
            def class_requires_deps(*deps):
                def _deco(cls): return cls
                return _deco
            deps_mod.class_requires_deps = class_requires_deps
            def function_requires_deps(*deps):
                def _deco(func): return func
                return _deco
            deps_mod.function_requires_deps = function_requires_deps

            print('[RuntimeHook] PaddleX deps patched OK')
        except Exception as e:
            print(f'[RuntimeHook] Could not patch paddlex deps: {e}')

    _patch_paddlex_deps()