import os
import sys
import subprocess
import argparse
import time
import numpy as np
try:
    from lxml import etree
except ImportError:
    pass

boolean_dict = \
    {'true' : True,   '1' : True,  'yes' : True,
     'false' : False, '0' : False, 'no' : False, 'none' : False }

# =============================================================================
# XML WRAPPERS
# =============================================================================

class ExtendableNamespace(argparse.Namespace):
    def AddNamespace(self,  **kwargs):
        for name in kwargs:
            att = getattr(self, name, None)
            if att is None:
                setattr(self, name, kwargs[name])
            else:
                setattr(self, name, kwargs[name].As(type(att)))
        return
    def Add(self, name, value):
        att = getattr(self, name, None)
        if att is None:
            setattr(self, name, value)
        else:
            att.Add(name, value)
        return value

def GenerateTreeDict(tree, element, path='', paths_rel_to=None):
    if type(element) == etree._Comment: return [], {}
    # Update path
    if path == '':
        if element.tag != paths_rel_to:
            path += element.tag
    else:
        path += '/' + element.tag
    # Containers for lower levels
    tag_node = {}
    nodes = []
    # Construct Node
    xmlnode = XmlNode(element, path) # tree.getpath(element))
    nodes.append(xmlnode)
    if len(element) == 0:
        tag_node[path] = xmlnode
    # Iterate over children
    for child in element:
        child_elements, childtag_element = GenerateTreeDict(tree, child, path)
        nodes = nodes + child_elements
        for key in childtag_element.keys():
            if tag_node.has_key(key):
                if type(tag_node[key]) != list:
                    tag_node[key] = [ tag_node[key], childtag_element[key] ]
                else:
                    tag_node[key].append(childtag_element[key])
            else:
                tag_node[key] = childtag_element[key]
    return nodes, tag_node

def NamespaceFromDict(tree_dict):
    nspace = ExtendableNamespace()
    for key in tree_dict.keys():
        sections = key.split('/')
        values = [ None for s in sections ]
        values[-1] = tree_dict[key]
        add_to_nspace = nspace
        for s,v in zip(sections, values):
            if v == None:
                if getattr(add_to_nspace, s, None):
                    add_to_nspace = getattr(add_to_nspace, s, None)
                else:
                    sub_nspace = ExtendableNamespace()
                    add_to_nspace = add_to_nspace.Add(s, sub_nspace)
            else:
                add_to_nspace.Add(s, v)
    return nspace

class XmlTree(list):
    def __init__(self, xmlfile, paths_rel_to=None):
        self.xmlfile = xmlfile
        self.xtree = etree.parse(xmlfile)
        self.xroot = self.xtree.getroot()
        self.nodes, self.tag_node = GenerateTreeDict(self.xtree, self.xroot, '', paths_rel_to)
        self.xspace = NamespaceFromDict(self.tag_node)
    def SelectByTag(self, tag):
        selection = [ e for e in self.nodes if e.tag == tag ]
        return selection
    def __getitem__(self, key):
        return self.tag_node[key]
    def keys(self):
        return self.tag_node.keys()

class XmlNode(object):
    def __init__(self, element, path):
        self.path = path
        self.node = element
        self.tag = element.tag        
        self.value = element.text
        self.attributes = element.attrib
    def As(self, typ):
        if typ == np.array:
            sps = self.value.split()
            return typ([ float(sp) for sp in sps ])
        elif typ == bool:
            return boolean_dict.get(self.value.lower())
        else:
            return typ(self.value)
    def AsArray(self, typ, sep=' ', rep='\t\n'):
        for r in rep:
            self.value = self.value.replace(r, sep)
        sp = self.value.split(sep)
        return [ typ(s) for s in sp if str(s) != '' ]
    def SetNodeValue(self, new_value):
        self.value = new_value
        if self.node != None:
            self.node.firstChild.nodeValue = new_value
        return
    def __getitem__(self, key):
        return self.node.get(key)

# =============================================================================
# COMMAND LINE & XML INPUT INTERFACE
# =============================================================================

class CLIO_HelpFormatter(argparse.HelpFormatter):
    def _format_usage(self, usage, action, group, prefix):
        return "%s : Command Line Interface\n" % sys.argv[0]

class OptionsInterface(object):
    def __init__(self):
        # COMMAND-LINE ARGUMENTS
        self.is_connected_to_cmd_ln = False
        self.cmd_ln_args = None
        self.cmd_ln_opts = None
        self.cmd_ln_nicknames = [ '-h' ]
        self.boolean_translator = boolean_dict
        self.subtype = str
        # XML OPTIONS FILE
        self.is_connected_to_xml = False
        self.xmlfile = None
        self.tree = None
        self.xdict = None
        self.xspace = None
        # JOINED OPTIONS
        self.opts = ExtendableNamespace()
    def Connect(self, xmlfile=None):
        self.ConnectToCmdLn()
        self.ConnectToOptionsFile(xmlfile)
    def Parse(self, xkey='options'):
        if self.is_connected_to_cmd_ln:
            self.ParseCmdLn()
        if self.is_connected_to_xml:
            self.ParseOptionsFileXml(xkey)
        if self.is_connected_to_cmd_ln and not self.is_connected_to_xml:
            return self.cmd_ln_opts
        elif self.is_connected_to_xml and not self.is_connected_to_cmd_ln:
            return self.xspace
        else:
            return self.cmd_ln_opts, self.xspace
    def ParseOptionsFile(self, xmlfile, xkey):
        self.xmlfile = xmlfile
        self.is_connected_to_xml = True
        self.ParseOptionsFileXml(xkey)
        return self.xspace
    # COMMAND-LINE PARSING
    def __call__(self):
        return self.cmd_ln_opts
    def ConnectToCmdLn(self, prog=sys.argv[0], descr=None):
        self.cmd_ln_args = argparse.ArgumentParser(prog=sys.argv[0],
            formatter_class=lambda prog: CLIO_HelpFormatter(prog,max_help_position=70))
        self.is_connected_to_cmd_ln = True
        return
    def ParseCmdLn(self):
        self.cmd_ln_opts = self.cmd_ln_args.parse_args()
    def InterpretAsBoolean(self, expr):
        try:
            return self.boolean_translator.get(expr.lower())
        except KeyError:
            raise ValueError('CLIO does not know how to convert %s into a boolean.' % expr)
    def InterpretAsNumpyArray(self, expr):
        array = [ float(e) for e in expr ]
        array = np.array(array)
        return array
    def InterpretAsList(self, expr):
        array = [ self.subtype(e) for e in expr ]
        return array
    def AddArg(self, name, typ=str, nickname=None, 
            default=None, destination=None, help=None):
        # Sort out <name> (e.g. --time) vs <destination> (e.g., time)
        if '--' != name[0:2]:
            dest = name
            name = '--' + name
        else:
            dest = name[2:]
        # Sort out <default> vs <required>
        if default == None: required = True
        else: required = False
        # Construct default <help> it not given
        if help == None: help = "%s <default: %s>" % (repr(type), repr(default))
        # Construct <nickname> if not given
        if nickname == None:
            nickname = '-'
            for char in dest:
                nickname += char
                if not nickname in self.cmd_ln_nicknames:
                    break
            if nickname in self.cmd_ln_nicknames:
                raise ValueError('CLIO could not construct nickname from %s option'\
                    % name)
            self.cmd_ln_nicknames.append(nickname)
        # Process type
        if typ in [int, float, str]:
            nargs = None
        elif typ == bool:
            typ = self.InterpretAsBoolean
            nargs = None
        elif typ == np.array:
            raise NotImplementedError
            typ = float # self.InterpretAsNumpyArray
            nargs = 3
        elif typ == list:
            typ = str
            nargs = '*'
        elif len(typ) == 2 and typ[0] == list:
            typ = typ[1]
            nargs = '*'
        else:
            raise NotImplementedError("CLIO does not know how to generate type '%s'"\
                 % typ)
        self.cmd_ln_args.add_argument(nickname, name, 
                dest=dest,
                action='store',
                nargs=nargs,
                required=required,
                type=typ,
                metavar=dest[0:1].upper(),
                default=default,
                help=help)
        return
    # OPTIONS FILE PARSING
    def ConnectToOptionsFile(self, xmlfile):
        if xmlfile == None or xmlfile == '': return
        self.xmlfile = xmlfile
        self.is_connected_to_xml = True
        return
    def ParseOptionsFileXml(self, xkey='options'):
        if self.xmlfile == None: return
        self.tree = XmlTree(self.xmlfile, paths_rel_to=xkey)
        self.xdict = self.tree.tag_node
        self.xspace = self.tree.xspace
        return
    def __getitem__(self, key):
        try:
            return self.xspace.__dict__[key]            
        except KeyError:
            return self.cmd_ln_opts.__dict__[key]
        except KeyError:
            raise AttributeError('No such option registered: \'%s\'' % key)
        return None
        
class ShellInterface(object):
    def __init__(self):
        # PRINTER ATTRIBUTES
        self.color_dict = { \
            'pp' : '\033[95m',
            'mb' : '\033[34m',
            'lb' : '\033[1;34m',
            'my' : '\033[1;33m',
            'mg' : '\033[92m',            
            'mr' : '\033[91m',
            'ww' : '\033[0;1m',
            'ok' : '\033[92m',
            'xx' : '\033[91m',
            'warning' : '\033[93m',
            'error' : '\033[95m',
            'endcolor' : '\033[0;1m' }
        self.justify_dict = { \
            'o' : '  o ',
            '.' : '... ',
            'r' : '\r',
            'ro' : '\r  o '}
        self.pp = OS_COLOR('pp')
        self.lb = OS_COLOR('lb')
        self.mb = OS_COLOR('mb')        
        self.mg = OS_COLOR('mg')
        self.my = OS_COLOR('my')        
        self.mr = OS_COLOR('mr')
        self.ww = OS_COLOR('ww')
        self.ok = OS_COLOR('ok')
        self.xx = OS_COLOR('xx')
        self.colors = [ OS_COLOR(c) for c in sorted(self.color_dict.keys()) ]
        self.item = '  o '
        self.iitem = '      - '
        self.endl = OS_LINE_CHAR('\n')
        self.flush = OS_LINE_CHAR('')
        self.back = OS_LINE_CHAR('\r')
        self.trail = ' '
        # CURRENT STYLE SELECTION
        self.sel_color = None
        self.sel_justify = None
        self.sel_header = False
        self.sel_trim = "="
        # EXE ATTRIBUTES
        self.catch = OS_EXE_CATCH()
        self.assert_zero = OS_EXE_ASSERT()
        self.dev = OS_EXE_DEV('')
        self.nodev = OS_EXE_DEV('')
        self.devnull = OS_EXE_DEV()
        self.devfile = OS_EXE_DEV('')
        self.os_exe_get = False
        self.os_exe_assert_zero = False
        self.os_exe_dev = ''
        self.os_exe_verbose = False
        # LOGGING
        self.logfile = None
        # DIRECTORY HOPPING
        self.paths_visited = [ os.getcwd() ]
        self.exe_root_path = self.paths_visited[0]
        self.N_store_paths_visited = 1+5
    def __call__(self, mssg, c=None, j=None, h=False, t="="):
        # c=color, j=justify, h=header, t=trim, u=upper-case
        if j:
            mssg = self.justify_dict[j] + mssg
        if c != None:
            mssg = self.color_dict[c] + mssg + self.color_dict['endcolor']
        if h:
            mssg = self.os_generate_header(mssg, t)
    # LOGFILE ADAPTOR
    def ConnectToFile(self, logfile):
        self.logfile = logfile
        sys.stdout = open(logfile, 'w')
        self.devfile = OS_EXE_DEV(' >> {log} 2>> {log}'.format(log=logfile))
        return
    def DisconnectFromFile(self):
        if self.logfile != None:
            self.devfile = OS_EXE_DEV('')
            self.logfile = None
            sys.stdout = sys.__stdout__
        else: pass
        return
    # PRINTER METHODS
    def __lshift__(self, mssg):
        if type(mssg) == OS_LINE_CHAR:
            # <LOG MESSAGE HERE>
            sys.stdout.write(str(mssg))
            sys.stdout.flush()
            self.sel_color = None
            return self
        elif type(mssg) == OS_COLOR:
            self.sel_color = str(mssg)
            return self
        mssg = str(mssg)
        if self.sel_justify != None:
            mssg = self.justify_dict[j] + mssg
        mssg += self.trail
        if self.sel_color != None:
            mssg = self.color_dict[self.sel_color] \
                + mssg + self.color_dict['endcolor']
        if self.sel_header:
            mssg = self.os_generate_header(mssg, self.sel_trim)
        # <LOG MESSAGE HERE>
        sys.stdout.write(mssg)
        return self
    def os_print(self, mssg, c=None, j=None, h=False, t="="):
        # c=color, j=justify, h=header, t=trim, u=upper-case
        if j:
            mssg = OS_JUSTIFY_DICT[j] + mssg
        if c != None:
            mssg = OS_COLOR_DICT[c] + mssg + OS_COLOR_DICT['endcolor']
        if h:
            mssg = os_generate_header(mssg, t)
        return
    def os_print_config(self, c=None, j=None, h=False, t="=", tl=' '):
        self.sel_color = c
        self.sel_justify = j
        self.sel_header = h
        self.sel_trim = t
        self.trail = tl
        return
    def os_print_reset(self):
        self.sel_color = None
        self.sel_justify = None
        self.sel_header = False
        self.sel_trim = "="
        self.trail = ' '
        return
    def os_generate_header(title, trim="="):
        try:
            height, width = os.popen('stty size', 'r').read().split()
            width = int(width)
            leftright = int((width - len(title)-2)/2)
        except ValueError:
            leftright = 40
        return trim*leftright + " " + title + " " + trim*leftright
    # SYSTEM COMMAND WRAPPER
    def __rshift__(self, cmmd):
        if type(cmmd) == OS_EXE_CATCH:
            self.os_exe_get = True
            return self
        elif type(cmmd) == OS_EXE_DEV:
            self.dev = cmmd
            return self
        elif type(cmmd) == OS_EXE_ASSERT:
            self.os_exe_assert_zero = True
            return self
        # Redirect command as requested
        if not self.os_exe_get:
            if str(self.dev) != '':
                cmmd += str(self.dev)
                self.dev = self.nodev
            else:
                cmmd += str(self.devfile)
        # Execute
        if self.debug: self << self.my << "exe:" << cmmd << endl
        if self.os_exe_get:
            output = subprocess.getoutput(cmmd)
            self.os_exe_get = False        
            return output
        else:
            sign = os.system(cmmd)
        if self.os_exe_assert_zero:
            if str(sign) != '0':
                raise RuntimeError("<OSIO> '%s' returned '%s'" % (cmmd, sign))
            self.os_exe_assert_zero = False
        return sign
    # PROGRAM EXIT
    def okquit(self, what=''):
        if what != '': self << self.ok << what << self.endl
        self.DisconnectFromFile()
        sys.exit(0)
        return
    def xxquit(self, what=''):
        if what != '': self << self.xx << "ERROR" << what << self.endl
        self.DisconnectFromFile()
        sys.exit(1)
        return
    # DIRECTORY NAVIGATION
    def cd(self, d):
        # Current working directory, for archiving ... =>
        cwd = os.getcwd()
        if type(d) == int:
            # Change to previously visited path
            os.chdir(self.paths_visited[d])
        elif type(d) == str:
            # Change to path as specified explicitly
            os.chdir(d)
        else:
            raise NotImplementedError
        # <= ... previous path
        self.paths_visited.append(cwd)
        if len(self.paths_visited) > self.N_store_paths_visited:
            self.paths_visited.pop(1) # 0 stores root
        if self.debug: self << self.my << "cd: " << os.getcwd() << self.endl
        return
    def pwd(self):
        return self.cwd()
    def cwd(self):
        return os.getcwd()
    def root(self):
        self.cd(self.exe_root_path)
        return
    def abspath(self, file):
        if not os.path.exists(file):
            raise IOError("<osio::abspath> No such item in local directory: '%s'" % file)
        return os.path.join(self.cwd(), file)
    def mkcd(self, directory):
        self >> self.assert_zero >> 'mkdir -p %s' % directory
        self.cd(directory)
        return directory

class OS_EXE_DEV(object):
    def __init__(self, dev=' > /dev/null 2> /dev/null'):
        self.dev = dev
    def __str__(self):
        return self.dev

class OS_EXE_CATCH(object):
    def __init__(self):
        self.catch = True
        
class OS_EXE_ASSERT(object):
    def __init__(self):
        self.assert_0 = True

class OS_COLOR(object):
    def __init__(self, colstr):
        self.colstr = colstr
    def __str__(self):
        return self.colstr

class OS_LINE_CHAR(object):
    def __init__(self, char):
        self.char = char
    def __str__(self):
        return self.char

class LOGGER(ShellInterface, OptionsInterface):
    def __init__(self):
        self.debug = False
        ShellInterface.__init__(self)
        OptionsInterface.__init__(self)
        return
    def sleep(self, dt):
        time.sleep(dt)
        return

log = LOGGER()
endl = OS_LINE_CHAR('\n')
flush = OS_LINE_CHAR('')
back = OS_LINE_CHAR('\r')
catch = OS_EXE_CATCH()
devnull = OS_EXE_DEV()
Mock = ExtendableNamespace

