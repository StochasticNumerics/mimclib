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
from IPython.core import compilerop
from IPython.terminal.ipapp import load_default_config

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
v(erbose): Verbose output of stack.
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
        self.tb_out = MyTB(color_scheme='Linux')
        # from IPython.core import ultratb as coreultratb
        # self.tb_verbose = coreultratb.VerboseTB(color_scheme='Linux')
        super(MyInteractiveShellEmbed, self).__init__(**kwargs)

    def init_magics(self):
        super(MyInteractiveShellEmbed, self).init_magics()
        self.register_magics(StackEmbeddedMagics)

    def __call__(self, ex, ex_cls, tb, frames, **kwargs):
        if not self.embedded_active:
            return

        self.frames = frames
        self.ex = ex
        self.tb = tb
        self.ex_cls = ex_cls
        self.cur_frame = len(self.frames)-1
        self.print_stack()

        self.cur_frame = len(self.frames)-1
        super(MyInteractiveShellEmbed,
              self).__call__(local_ns=self.frames[-1].f_locals,
                             module=sys.modules[self.frames[-1].f_globals['__name__']],
                             global_ns=self.frames[-1].f_globals,
                             compile_flags=self.frames[-1].f_code.co_flags & compilerop.PyCF_MASK,**kwargs)

    def print_verbose_stack(self):
        #self.tb_verbose(self.ex_cls, self.ex, self.tb)
        return self.tb_out.print_verbose(etype=self.ex_cls,
                                         evalue=self.ex,
                                         frames=self.frames)

    def print_stack(self):
        return self.tb_out.print_list(etype=self.ex_cls,
                                      evalue=self.ex,
                                      frames=self.frames,
                                      hightlight=self.cur_frame)

@public
def embed(**kwargs):
    stack_depth = kwargs.pop('stack_depth', 2)
    force_active = kwargs.pop('force_active', False)
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
        i = 0
        while True:
            try:
                frames.append(sys._getframe(i).f_back)
                i += 1
            except ValueError:
                break
        #frames = frames[-2::-1]
        frames = frames[:-1]
        frames = frames[::-1]

    tb = kwargs.pop("tb", None)
    old_excepthool = sys.excepthook
    shell = MyInteractiveShellEmbed.instance(**kwargs)
    if force_active:
        shell.embedded_active = True
    shell(frames=frames, ex_cls=ex_cls, ex=ex, tb=tb, header=header,
          stack_depth=stack_depth)
    sys.excepthook = old_excepthool


def __excepthook(ex_cls, ex, tb):
    frames = []
    tb_cur = tb
    while tb_cur is not None:
        frames.append(tb_cur.tb_frame)
        tb_cur = tb_cur.tb_next
    embed(ex_cls=ex_cls, ex=ex, frames=frames, tb=tb, force_active=True)

@public
def set_excepthook():
    sys.excepthook = __excepthook


######################################################################
######################################################################
# Code for outputting of traceback/frames

import inspect
import keyword
import os
import tokenize
import types
import sys
import time
from IPython.core import ultratb
from IPython.utils import py3compat
from IPython.utils import path as util_path
from IPython.utils import ulinecache
from IPython.utils.data import uniq_stable
from IPython.utils import PyColorize

try:                           # Python 2
    generate_tokens = tokenize.generate_tokens
except AttributeError:         # Python 3
    generate_tokens = tokenize.tokenize

_parser = PyColorize.Parser()
def _format_traceback_lines(lnum, index, lines, Colors, lvals=None,scheme=None):
    numbers_width = ultratb.INDENT_SIZE - 1
    res = []
    i = lnum - index

    # This lets us get fully syntax-highlighted tracebacks.
    if scheme is None:
        ipinst = get_ipython()
        if ipinst is not None:
            scheme = ipinst.colors
        else:
            scheme = DEFAULT_SCHEME

    _line_format = _parser.format2

    for line in lines:
        line = py3compat.cast_unicode(line)

        new_line, err = _line_format(line, 'str', scheme)
        if not err: line = new_line

        if i == lnum:
            # This is the line with the error
            pad = numbers_width - len(str(i))
            if pad >= 3:
                marker = '-'*(pad-3) + '-> '
            elif pad == 2:
                marker = '> '
            elif pad == 1:
                marker = '>'
            else:
                marker = ''
            num = marker + str(i)
            line = '%s%s%s %s%s' %(Colors.linenoEm, num,
                                   Colors.line, line, Colors.Normal)
        else:
            num = '%*s' % (numbers_width,i)
            line = '%s%s%s %s' %(Colors.lineno, num,
                                 Colors.Normal, line)

        res.append(line)
        if lvals and i == lnum:
            res.append(lvals + '\n')
        i = i + 1
    return res

class MyTB(ultratb.ListTB):
    def __init__(self, color_scheme='NoColor', call_pdb=False,
                 ostream=None):
        super(MyTB, self).__init__(color_scheme=color_scheme,
                                   call_pdb=call_pdb, ostream=ostream)

    def _output_list(self,out_list, sep=''):
        outStr = sep.join(out_list)
        self.ostream.flush()
        self.ostream.write(outStr)
        self.ostream.flush()
        return outStr

    def print_list(self, frames, hightlight=None, evalue=None,
                   etype=None):
        if hightlight is None:
            hightlight = len(frames)-1
        Colors = self.Colors
        out_list = []
        out_list.append('Traceback %s(most recent call last)%s:' %
                        (Colors.normalEm, Colors.Normal) + '\n')
        out_list.extend(self._format_frame_list(frames, hightlight))
        # The exception info should be a single entry in the list.
        if evalue is not None and etype is not None:
            lines = ''.join(self._format_exception_only(etype, evalue))
            out_list.append(lines)
        return self._output_list(out_list)

    def _format_frame_list(self, frames, hightlight):
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
        for i, frm in enumerate(frames):
            filename, lineno, name, lines, linei = inspect.getframeinfo(frm)
            line = lines[linei] if lines is not None else None
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


    def print_verbose(self, frames, evalue=None, etype=None,
                      context=5, long_header=False, include_vars=True):
        """Return a nice text document describing the traceback."""
        # some locals
        try:
            etype = etype.__name__
        except AttributeError:
            pass

        Colors        = self.Colors   # just a shorthand + quicker name lookup
        ColorsNormal  = Colors.Normal  # used a lot
        col_scheme    = self.color_scheme_table.active_scheme_name
        indent        = ' '*ultratb.INDENT_SIZE
        em_normal     = '%s\n%s%s' % (Colors.valEm, indent,ColorsNormal)
        undefined     = '%sundefined%s' % (Colors.em, ColorsNormal)
        exc = '%s%s%s' % (Colors.excName,etype,ColorsNormal)

        # some internal-use functions
        def text_repr(value):
            """Hopefully pretty robust repr equivalent."""
            # this is pretty horrible but should always return *something*
            try:
                return pydoc.text.repr(value)
            except KeyboardInterrupt:
                raise
            except:
                try:
                    return repr(value)
                except KeyboardInterrupt:
                    raise
                except:
                    try:
                        # all still in an except block so we catch
                        # getattr raising
                        name = getattr(value, '__name__', None)
                        if name:
                            # ick, recursion
                            return text_repr(name)
                        klass = getattr(value, '__class__', None)
                        if klass:
                            return '%s instance' % text_repr(klass)
                    except KeyboardInterrupt:
                        raise
                    except:
                        return 'UNRECOVERABLE REPR FAILURE'
        def eqrepr(value, repr=text_repr): return '=%s' % repr(value)
        def nullrepr(value, repr=text_repr): return ''

        # meat of the code begins
        if long_header:
            # Header with the exception type, python version, and date
            pyver = 'Python ' + sys.version.split()[0] + ': ' + sys.executable
            date = time.ctime(time.time())

            if etype is not None:
                head = '%s%s%s\n%s%s%s\n%s' % (Colors.topline, '-'*75, ColorsNormal,
                                               exc, ' '*(75-len(str(etype))-len(pyver)),
                                               pyver, date.rjust(75) )
                head += "\nA problem occured executing Python code.  Here is the sequence of function"\
                        "\ncalls leading up to the error, with the most recent (innermost) call last."
            else:
                head = '%s%s\n%s%s%s\n%s' % (Colors.topline, '-'*75, ColorsNormal,
                                               ' '*(75-len(pyver)),
                                               pyver, date.rjust(75) )
        else:
            # Simplified header
            if etype is None:
                head = '%s%s\n%s%s' % (Colors.topline, '-'*75, ColorsNormal,
                                         'Traceback (most recent call last)' )
            else:
                head = '%s%s%s\n%s%s' % (Colors.topline, '-'*75, ColorsNormal,exc,
                                         'Traceback (most recent call last)'.\
                                         rjust(75 - len(str(etype)) ) )
        frames_txt = []

        # build some color string templates outside these nested loops
        tpl_link       = '%s%%s%s' % (Colors.filenameEm,ColorsNormal)
        tpl_call       = 'in %s%%s%s%%s%s' % (Colors.vName, Colors.valEm,
                                              ColorsNormal)
        tpl_call_fail  = 'in %s%%s%s(***failed resolving arguments***)%s' % \
                         (Colors.vName, Colors.valEm, ColorsNormal)
        tpl_local_var  = '%s%%s%s' % (Colors.vName, ColorsNormal)
        tpl_global_var = '%sglobal%s %s%%s%s' % (Colors.em, ColorsNormal,
                                                 Colors.vName, ColorsNormal)
        tpl_name_val   = '%%s %s= %%s%s' % (Colors.valEm, ColorsNormal)
        tpl_line       = '%s%%s%s %%s' % (Colors.lineno, ColorsNormal)
        tpl_line_em    = '%s%%s%s %%s%s' % (Colors.linenoEm,Colors.line,
                                            ColorsNormal)

        # now, loop over all records printing context and info
        abspath = os.path.abspath
        for frame in frames:
            file, lnum, func, lines, index = inspect.getframeinfo(frame,
                                                                  context=context)
            #print '*** record:',file,lnum,func,lines,index  # dbg
            if not file:
                file = '?'
            elif not(file.startswith(str("<")) and file.endswith(str(">"))):
                # Guess that filenames like <string> aren't real filenames, so
                # don't call abspath on them.
                try:
                    file = abspath(file)
                except OSError:
                    # Not sure if this can still happen: abspath now works with
                    # file names like <string>
                    pass
            file = py3compat.cast_unicode(file, util_path.fs_encoding)
            link = tpl_link % file
            args, varargs, varkw, locals = inspect.getargvalues(frame)

            if func == '?':
                call = ''
            else:
                # Decide whether to include variable details or not
                var_repr = include_vars and eqrepr or nullrepr
                try:
                    call = tpl_call % (func,inspect.formatargvalues(args,
                                                varargs, varkw,
                                                locals,formatvalue=var_repr))
                except KeyError:
                    # This happens in situations like errors inside generator
                    # expressions, where local variables are listed in the
                    # line, but can't be extracted from the frame.  I'm not
                    # 100% sure this isn't actually a bug in inspect itself,
                    # but since there's no info for us to compute with, the
                    # best we can do is report the failure and move on.  Here
                    # we must *not* call any traceback construction again,
                    # because that would mess up use of %debug later on.  So we
                    # simply report the failure and move on.  The only
                    # limitation will be that this frame won't have locals
                    # listed in the call signature.  Quite subtle problem...
                    # I can't think of a good way to validate this in a unit
                    # test, but running a script consisting of:
                    #  dict( (k,v.strip()) for (k,v) in range(10) )
                    # will illustrate the error, if this exception catch is
                    # disabled.
                    call = tpl_call_fail % func

            # Don't attempt to tokenize binary files.
            if file.endswith(('.so', '.pyd', '.dll')):
                frames_txt.append('%s %s\n' % (link,call))
                continue
            elif file.endswith(('.pyc','.pyo')):
                # Look up the corresponding source file.
                file = openpy.source_from_cache(file)

            def linereader(file=file, lnum=[lnum], getline=ulinecache.getline):
                line = getline(file, lnum[0])
                lnum[0] += 1
                return line

            # Build the list of names on this line of code where the exception
            # occurred.
            try:
                names = []
                name_cont = False

                for token_type, token, start, end, line in generate_tokens(linereader):
                    # build composite names
                    if token_type == tokenize.NAME and token not in keyword.kwlist:
                        if name_cont:
                            # Continuation of a dotted name
                            try:
                                names[-1].append(token)
                            except IndexError:
                                names.append([token])
                            name_cont = False
                        else:
                            # Regular new names.  We append everything, the caller
                            # will be responsible for pruning the list later.  It's
                            # very tricky to try to prune as we go, b/c composite
                            # names can fool us.  The pruning at the end is easy
                            # to do (or the caller can print a list with repeated
                            # names if so desired.
                            names.append([token])
                    elif token == '.':
                        name_cont = True
                    elif token_type == tokenize.NEWLINE:
                        break

            except (IndexError, UnicodeDecodeError):
                # signals exit of tokenizer
                pass
            except tokenize.TokenError as msg:
                _m = ("An unexpected error occurred while tokenizing input\n"
                      "The following traceback may be corrupted or invalid\n"
                      "The error message is: %s\n" % msg)
                error(_m)

            # Join composite names (e.g. "dict.fromkeys")
            names = ['.'.join(n) for n in names]
            # prune names list of duplicates, but keep the right order
            unique_names = uniq_stable(names)

            # Start loop over vars
            lvals = []
            if include_vars:
                for name_full in unique_names:
                    name_base = name_full.split('.',1)[0]
                    if name_base in frame.f_code.co_varnames:
                        if name_base in locals:
                            try:
                                value = repr(eval(name_full,locals))
                            except:
                                value = undefined
                        else:
                            value = undefined
                        name = tpl_local_var % name_full
                    else:
                        if name_base in frame.f_globals:
                            try:
                                value = repr(eval(name_full,frame.f_globals))
                            except:
                                value = undefined
                        else:
                            value = undefined
                        name = tpl_global_var % name_full
                    lvals.append(tpl_name_val % (name,value))
            if lvals:
                lvals = '%s%s' % (indent,em_normal.join(lvals))
            else:
                lvals = ''

            level = '%s %s\n' % (link,call)

            if index is None:
                frames_txt.append(level)
            else:
                frames_txt.append('%s%s' % (level,''.join(
                    _format_traceback_lines(lnum,index,lines,Colors,lvals,
                                            col_scheme))))

        # Get (safely) a string form of the exception info
        if evalue is not None and etype is not None:
            try:
                etype_str,evalue_str = map(str,(etype,evalue))
            except:
                # User exception is improperly defined.
                etype,evalue = str,sys.exc_info()[:2]
                etype_str,evalue_str = map(str,(etype,evalue))
            # ... and format it
            exception = ['%s%s%s: %s' % (Colors.excName, etype_str,
                                         ColorsNormal, py3compat.cast_unicode(evalue_str))]
            if (not py3compat.PY3) and type(evalue) is types.InstanceType:
                try:
                    names = [w for w in dir(evalue) if isinstance(w, py3compat.string_types)]
                except:
                    # Every now and then, an object with funny inernals blows up
                    # when dir() is called on it.  We do the best we can to report
                    # the problem and continue
                    _m = '%sException reporting error (object with broken dir())%s:'
                    exception.append(_m % (Colors.excName,ColorsNormal))
                    etype_str,evalue_str = map(str,sys.exc_info()[:2])
                    exception.append('%s%s%s: %s' % (Colors.excName,etype_str,
                                         ColorsNormal, py3compat.cast_unicode(evalue_str)))
                    names = []
                for name in names:
                    value = text_repr(getattr(evalue, name))
                    exception.append('\n%s%s = %s' % (indent, name, value))
            exp_list = [''.join(exception[0]), '\n']
        else:
            exp_list = []
        # vds: >>
        if frames:
            filepath, lnum, _, _, _ = inspect.getframeinfo(frames[-1])
            filepath = os.path.abspath(filepath)
            ipinst = get_ipython()
            if ipinst is not None:
                ipinst.hooks.synchronize_with_editor(filepath, lnum, 0)
        # vds: <<

        # return all our info assembled as a single string
        # return '%s\n\n%s\n%s' % (head,'\n'.join(frames_txt),''.join(exception[0]) )
        return self._output_list([head] + frames_txt + exp_list, sep='\n')
