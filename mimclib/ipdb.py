# encoding: utf-8
"""
An embedded IPython shell.

Author:

* Abdul-Lateef Haji-Ali

Notes
-----
Some parts are inspired/copied from IPython.embed and IPython.core.ultratb
"""

#-----------------------------------------------------------------------------
#  Copyright (C) 2008-2011  The IPython Development Team
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function

__all__ = []

def public(sym):
    __all__.append(sym.__name__)
    return sym


import sys
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.terminal.embed import InteractiveShellEmbed
from IPython.core import ultratb
from IPython.core import compilerop
from IPython.terminal.ipapp import load_default_config
import inspect

@magics_class
class StackEmbeddedMagics(Magics):
    @line_magic
    def stack(self, parameter_s=''):
        # parameter_s: +, - or number
        parameter_s = parameter_s.lower().strip()
        if len(parameter_s) == 0:
            self.shell.print_stack()
            return
        if parameter_s.startswith("h") or parameter_s.startswith("?"):
            print('''u(p):       Stack up.
d(own):     Stack down.
v(verbose): Verbose output of stack.
2:          Jump to stack number 2 (or any other number)''')
            return
        if parameter_s.startswith("v"):
            self.shell.print_verbose_stack()
            return

        cur_frame = self.shell.cur_frame
        if parameter_s.startswith("u"):
            cur_frame -= 1
        elif parameter_s.startswith("d"):
            cur_frame += 1
        else:
            try:
                cur_frame = int(parameter_s)
                if cur_frame < 0:
                    cur_frame += len(self.shell.frames)
            except ValueError:
                print("ERROR: Must pass nothing, 'v', 'u', 'd' or a number. Nothing changed.")
                return

        if cur_frame < 0 or cur_frame >= len(self.shell.frames):
            print("ERROR: Frame out of bounds. Nothing changed")
            return
        if cur_frame != self.shell.cur_frame:
            # Copy old ipython locals
            new_local = dict()
            frm = self.shell.frames[self.shell.cur_frame]
            for k, v in self.shell.user_ns.iteritems():
                if k not in frm.f_locals:
                    new_local[k] = v

            self.shell.cur_frame = cur_frame
            frm = self.shell.frames[self.shell.cur_frame]
            for k, v in frm.f_locals.iteritems():
                new_local[k] = v

            self.shell.user_module = sys.modules[frm.f_globals['__name__']]
            self.shell.user_ns = new_local
            self.shell.init_user_ns()
            self.shell.set_completer_frame()
        self.shell.print_stack()


class MyInteractiveShellEmbed(InteractiveShellEmbed):
    def __init__(self, **kwargs):
        self.list_tb = ultratb.ListTB(color_scheme='Linux')
        self.verbose_tb = ultratb.VerboseTB(color_scheme='Linux')

        super(MyInteractiveShellEmbed, self).__init__(**kwargs)

    def init_magics(self):
        super(MyInteractiveShellEmbed, self).init_magics()
        self.register_magics(StackEmbeddedMagics)

    def __call__(self, ex, ex_cls, tb, frames, **kwargs):
        self.frames = frames
        self.ex = ex
        self.tb = tb
        self.ex_cls = ex_cls
        self.cur_frame = len(self.frames)-1
        self.print_stack(self.cur_frame)

        self.cur_frame = len(self.frames)-1
        super(MyInteractiveShellEmbed,
              self).__call__(local_ns=self.frames[-1].f_locals,
                             module=sys.modules[self.frames[-1].f_globals['__name__']],
                             global_ns=self.frames[-1].f_globals,
                             compile_flags=self.frames[-1].f_code.co_flags & compilerop.PyCF_MASK,**kwargs)

    def print_verbose_stack(self, hightlight=None):
        self.verbose_tb(self.ex_cls, self.ex, self.tb)

    def print_stack(self, hightlight=None):
        hightlight = hightlight or self.cur_frame
        Colors = self.list_tb.Colors
        out_list = []
        out_list.append('Traceback %s(most recent call last)%s:' %
                        (Colors.normalEm, Colors.Normal) + '\n')
        out_list.extend(self._format_frame_list(hightlight))
        # The exception info should be a single entry in the list.
        if self.ex is not None:
            lines = ''.join(self.list_tb._format_exception_only(self.ex_cls,
                                                                self.ex))
            out_list.append(lines)

        self.list_tb.ostream.flush()
        self.list_tb.ostream.write(''.join(out_list))
        self.list_tb.ostream.flush()
        return ''.join(out_list)

    def _format_frame_list(self, hightlight):
        """Format a list of traceback entry tuples for printing.

        Given a list of tuples as returned by extract_tb() or
        extract_stack(), return a list of strings ready for printing.
        Each string in the resulting list corresponds to the item with the
        same index in the argument list.  Each string ends in a newline;
        the strings may contain internal newlines as well, for those items
        whose source text line is not None.

        Lifted almost verbatim from traceback.py
        """
        Colors = self.list_tb.Colors
        list = []
        for i, frm in enumerate(self.frames):
            filename, lineno, name, lines, linei = inspect.getframeinfo(frm)
            line = lines[linei]
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

@public
def embed(**kwargs):
    stack_depth = kwargs.pop('stack_depth', 2)
    ex_data = kwargs.pop('ex_data', None)
    config = kwargs.get('config')
    header = kwargs.pop('header', u'')
    if config is None:
        config = load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        kwargs['config'] = config
    if "display_banner" not in kwargs:
        kwargs["display_banner"] = False

    ex_cls = kwargs.pop("ex_cls", None)
    ex = kwargs.pop("ex", None)
    if "frames" in kwargs:
        frames = kwargs.pop("frames")
    else:
        # Get from local
        frames = []
        i = 1
        while True:
            try:
                frames.append(sys._getframe(i).f_back)
                i += 1
            except ValueError:
                break
        frames = frames[::-1]

    tb = kwargs.pop("tb", _FauxTb.gen_tb(frames))
    tb = _FauxTb.gen_tb(frames)
    old_excepthool = sys.excepthook
    shell = MyInteractiveShellEmbed.instance(**kwargs)
    shell(frames=frames, ex_cls=ex_cls, ex=ex, tb=tb, header=header,
          stack_depth=stack_depth)
    sys.excepthook = old_excepthool


class _FauxTb():
    def __init__(self):
        self.tb_frame = None
        self.f_lineno = self.tb_lineno = None
        self.tb_next = None

    @staticmethod
    def gen_tb(frames):
        prev = None
        start = None
        for frm in frames:
            cur = _FauxTb()
            if prev:
                prev.tb_next = cur
            else:
                start = cur
            filename, lineno, name, lines, linei = inspect.getframeinfo(frm)
            cur.tb_frame = frm
            cur.f_lineno = cur.tb_lineno = linei
            prev = cur
        return start

def __excepthook(ex_cls, ex, tb):
    frames = []
    tb_cur = tb
    while tb_cur is not None:
        frames.append(tb_cur.tb_frame)
        tb_cur = tb_cur.tb_next
    embed(ex_cls=ex_cls, ex=ex, frames=frames, tb=tb)

@public
def set_excepthook():
    sys.excepthook = __excepthook
