from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym

skip_embed = False

@public
def embed(**kwargs):
    from IPython.terminal.ipapp import load_default_config
    from IPython.terminal.embed import InteractiveShellEmbed

    if kwargs.pop('skippable', True) and skip_embed:
        return

    stack_depth = kwargs.pop('stack_depth', 2)
    local_ns = kwargs.pop('local_ns', None)
    global_ns = kwargs.pop('global_ns', None)
    module = kwargs.pop('module', None)

    config = kwargs.get('config')
    header = kwargs.pop('header', u'')
    compile_flags = kwargs.pop('compile_flags', None)
    if config is None:
        config = load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs['config'] = config

    import sys
    old_excepthool = sys.excepthook
    shell = InteractiveShellEmbed.instance(display_banner=False, **kwargs)
    shell(header=header, stack_depth=stack_depth, compile_flags=compile_flags,
          local_ns=local_ns, global_ns=global_ns, module=module)
    sys.excepthook = old_excepthool

@public
class excepthook(object):
    def __init__(self):
        from IPython.core import ultratb
        from IPython.core.compilerop import check_linecache_ipython
        self.print_stack = ultratb.FormattedTB(color_scheme='Linux')

    def __call__(self, ex_cls, ex, tb):
        import sys
        from IPython.core import compilerop
        self.print_stack(ex_cls, ex, tb)

        tb_cur = tb
        while tb_cur is not None:
            frame = tb_cur.tb_frame
            tb_cur = tb_cur.tb_next
        embed(skippable=False, local_ns=frame.f_locals,
              module=sys.modules[frame.f_globals['__name__']],
              global_ns=frame.f_globals,
              compile_flags=(frame.f_code.co_flags &
                             compilerop.PyCF_MASK))
