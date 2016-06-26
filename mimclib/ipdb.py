from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym

from IPython.core.magic import Magics, magics_class, line_magic
from IPython.terminal.embed import InteractiveShellEmbed

@magics_class
class StackEmbeddedMagics(Magics):
    @line_magic
    def stack(self, parameter_s=''):
        # parameter_s: +, - or number
        if self.shell.frames is None:
            print("Stack is empty")
            return
        parameter_s = parameter_s.lower().strip()
        if len(parameter_s) == 0:
            self.shell.print_stack()
            return
        if parameter_s.startswith("v"):
            self.shell.print_verbose_stack()
            return

        if parameter_s.startswith("u"):
            self.shell.cur_frame -= 1
        elif parameter_s.startswith("d"):
            self.shell.cur_frame += 1
        else:
            try:
                self.shell.cur_frame = int(parameter_s)
            except ValueError:
                print("Must pass 'u' or 'd' or a number.")
                return

        self.shell.cur_frame = min(self.shell.cur_frame,
                                   len(self.shell.frames)-1)
        self.shell.cur_frame = max(self.shell.cur_frame, 0)

        # TODO: Copy old stuff that was saved
        import sys
        cur_frame = self.shell.frames[self.shell.cur_frame]
        self.shell.user_module = sys.modules[cur_frame.f_globals['__name__']]
        self.shell.user_ns = cur_frame.f_locals
        self.shell.init_user_ns()
        self.shell.set_completer_frame()

        self.shell.print_stack()


class MyInteractiveShellEmbed(InteractiveShellEmbed):
    def __init__(self, **kwargs):
        from IPython.core import ultratb
        from IPython.core.compilerop import check_linecache_ipython

        import sys
        super(MyInteractiveShellEmbed, self).__init__(**kwargs)
        self.ex_data = None
        self.frames = None
        self.cur_frame = None
        self.verbose_tb = ultratb.VerboseTB(color_scheme='Linux',
                                            include_vars=True)

        from IPython.core.excolors import exception_colors
        self.color_scheme_table = exception_colors()
        self.color_scheme_table.set_active_scheme('Linux')
        # for convenience, set Colors to the active scheme
        self.Colors = self.color_scheme_table.active_colors
        from IPython.utils import io
        self.ostream = io.stdout

    def init_magics(self):
        super(MyInteractiveShellEmbed, self).init_magics()
        self.register_magics(StackEmbeddedMagics)

    def __call__(self, **kwargs):
        if self.ex_data is not None:
            self.print_stack(self.cur_frame)
        super(MyInteractiveShellEmbed, self).__call__(**kwargs)

    def print_verbose_stack(self):
        self.verbose_tb(*self.ex_data)

    def print_stack(self, hightlight=None):
        etype, value, tb = self.ex_data
        import traceback
        elist = traceback.extract_tb(tb)
        hightlight = hightlight or self.cur_frame
        Colors = self.Colors
        out_list = []
        if elist:
            out_list.append('Traceback %s(most recent call last)%s:' %
                                (Colors.normalEm, Colors.Normal) + '\n')
            out_list.extend(self._format_list(elist, hightlight))
        # The exception info should be a single entry in the list.
        lines = ''.join(self._format_exception_only(etype, value))
        out_list.append(lines)

        self.ostream.flush()
        self.ostream.write(''.join(out_list))
        self.ostream.flush()
        return ''.join(out_list)

    def _format_list(self, extracted_list, hightlight):
        """Format a list of traceback entry tuples for printing.

        Given a list of tuples as returned by extract_tb() or
        extract_stack(), return a list of strings ready for printing.
        Each string in the resulting list corresponds to the item with the
        same index in the argument list.  Each string ends in a newline;
        the strings may contain internal newlines as well, for those items
        whose source text line is not None.

        Lifted almost verbatim from traceback.py
        """
        Colors = self.Colors
        list = []
        for i, dd in enumerate(extracted_list):
            filename, lineno, name, line = dd
            if i == hightlight:
                item = '%s  File %s"%s"%s, line %s%d%s, in %s%s%s%s\n' % \
                (Colors.normalEm,
                 Colors.filenameEm, filename, Colors.normalEm,
                 Colors.linenoEm, lineno, Colors.normalEm,
                 Colors.nameEm, name, Colors.normalEm,
                 Colors.Normal)
                if line:
                    item += '%s    %s%s\n' % (Colors.line, line.strip(),
                                              Colors.Normal)
            else:
                item = '  File %s"%s"%s, line %s%d%s, in %s%s%s\n' % \
                    (Colors.filename, filename, Colors.Normal,
                     Colors.lineno, lineno, Colors.Normal,
                     Colors.name, name, Colors.Normal)
                if line:
                    item += '    %s\n' % line.strip()
            list.append(item)
        return list

    def _format_exception_only(self, etype, value):
        """Format the exception part of a traceback.

        The arguments are the exception type and value such as given by
        sys.exc_info()[:2]. The return value is a list of strings, each ending
        in a newline.  Normally, the list contains a single string; however,
        for SyntaxError exceptions, it contains several lines that (when
        printed) display detailed information about where the syntax error
        occurred.  The message indicating which exception occurred is the
        always last string in the list.

        Also lifted nearly verbatim from traceback.py
        """
        have_filedata = False
        Colors = self.Colors
        list = []
        stype = Colors.excName + etype.__name__ + Colors.Normal
        if value is None:
            # Not sure if this can still happen in Python 2.6 and above
            list.append( py3compat.cast_unicode(stype) + '\n')
        else:
            if issubclass(etype, SyntaxError):
                have_filedata = True
                #print 'filename is',filename  # dbg
                if not value.filename: value.filename = "<string>"
                if value.lineno:
                    lineno = value.lineno
                    textline = ulinecache.getline(value.filename, value.lineno)
                else:
                    lineno = 'unknown'
                    textline = ''
                list.append('%s  File %s"%s"%s, line %s%s%s\n' % \
                        (Colors.normalEm,
                         Colors.filenameEm, py3compat.cast_unicode(value.filename), Colors.normalEm,
                         Colors.linenoEm, lineno, Colors.Normal  ))
                if textline == '':
                    textline = py3compat.cast_unicode(value.text, "utf-8")

                if textline is not None:
                    i = 0
                    while i < len(textline) and textline[i].isspace():
                        i += 1
                    list.append('%s    %s%s\n' % (Colors.line,
                                                  textline.strip(),
                                                  Colors.Normal))
                    if value.offset is not None:
                        s = '    '
                        for c in textline[i:value.offset-1]:
                            if c.isspace():
                                s += c
                            else:
                                s += ' '
                        list.append('%s%s^%s\n' % (Colors.caret, s,
                                                   Colors.Normal) )

            try:
                s = value.msg
            except Exception:
                s = self._some_str(value)
            if s:
                list.append('%s%s:%s %s\n' % (str(stype), Colors.excName,
                                              Colors.Normal, s))
            else:
                list.append('%s\n' % str(stype))

        # sync with user hooks
        if have_filedata:
            ipinst = get_ipython()
            if ipinst is not None:
                ipinst.hooks.synchronize_with_editor(value.filename, value.lineno, 0)

        return list

    def _some_str(self, value):
        # Lifted from traceback.py
        try:
            return str(value)
        except:
            return '<unprintable %s object>' % type(value).__name__


@public
def embed(**kwargs):
    from IPython.terminal.ipapp import load_default_config
    import sys

    stack_depth = kwargs.pop('stack_depth', 2)
    local_ns = kwargs.pop('local_ns', None)
    global_ns = kwargs.pop('global_ns', None)
    module = kwargs.pop('module', None)
    compile_flags = kwargs.pop('compile_flags', None)

    ex_data = kwargs.pop('ex_data', None)
    if ex_data is not None:
        ex_cls, ex, tb = ex_data
        frames = []
        tb_cur = tb
        while tb_cur is not None:
            frames.append(tb_cur.tb_frame)
            tb_cur = tb_cur.tb_next

        from IPython.core import compilerop
        local_ns = local_ns or frames[-1].f_locals
        module = module or sys.modules[frames[-1].f_globals['__name__']]
        global_ns = global_ns or frames[-1].f_globals
        compile_flags = compile_flags or (frames[-1].f_code.co_flags & compilerop.PyCF_MASK)

    config = kwargs.get('config')
    header = kwargs.pop('header', u'')
    if config is None:
        config = load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs['config'] = config

    old_excepthool = sys.excepthook
    shell = MyInteractiveShellEmbed.instance(display_banner=False, **kwargs)
    shell.ex_data = ex_data
    shell.frames = frames
    if frames is not None:
        shell.cur_frame = len(frames)-1
    shell(header=header, stack_depth=stack_depth, compile_flags=compile_flags,
          local_ns=local_ns, global_ns=global_ns, module=module)
    sys.excepthook = old_excepthool

@public
class excepthook(object):
    def __call__(self, ex_cls, ex, tb):
        embed(ex_data=[ex_cls, ex, tb])
