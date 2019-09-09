
class CommandNotExecuted(Exception):

    def __init__(self, cmd_name):
        super().__init__(cmd_name+" not executed")
        self.cmd_name = cmd_name

    def __repr__(self):
        return "'"+self.cmd_name+"' command is not executed"


class Command:

    def __init__(self, command_name, description, inf_positional, function):
        self.description = description
        self.command_name = command_name
        self.positional_arguments = dict()
        self.compulsory_arguments = dict()
        self.optional_arguments = dict()
        self.function = function

        self.inf_positional = inf_positional
        self.has_positional = False
        self.has_optional = True
        self.has_compulsory = False

    def __repr__(self):
        string = "usage: " + self.command_name
        if self.has_optional:
            for v in self.optional_arguments.values():
                string += " [" + v['sh'] + "]"

        if self.has_compulsory:
            for v in self.compulsory_arguments.values():
                string += " "+v['sh']

        if self.has_positional:
            for v in self.positional_arguments.values():
                string += " "+v['sh']
        if self.inf_positional:
            string += " ..."

        return string

    def show_help(self, out_func=print):
        string = self.__repr__()
        string += "\n\n"
        string += self.description + "\n"
        string += "\n"
        if self.has_positional:
            string += "positional arguments (all compulsory):\n"
            for v in self.positional_arguments.values():
                string += "\t" + v['sh'] + "\t" + str(v['type']).replace('<class ', "").replace(">", "") + "\t" + v[
                    'lf'] + "\t" + v['des'] + "\n"

        if self.has_compulsory:
            string += "compulsory arguments with options:\n"
            for v in self.compulsory_arguments.values():
                string += "\t" + v['sh'] + "\t" + str(v['type']).replace('<class ', "").replace(">", "") + "\t" + v[
                    'lf'] + "\t" + v['des'] + "\n"

        if self.has_optional:
            string += "optional arguments with options:\n"
            for v in self.optional_arguments.values():
                string += "\t" + v['sh'] + "\t" + str(v['type']).replace('<class ', "").replace(">", "") + "\t" + v[
                    'lf'] + "\t" + v['des'] + "\n"
        if self.inf_positional:
            string += "Infinite positional parameters\n"
        out_func(string)

    def add_positional_arguments(self, position, short_form, long_form, description, param_type=str):
        self.has_positional = True
        self.positional_arguments[position] = dict()
        self.positional_arguments[position]['sh'] = short_form
        self.positional_arguments[position]['lf'] = long_form
        self.positional_arguments[position]['des'] = description
        self.positional_arguments[position]['type'] = param_type

    def add_optional_arguments(self, short_form, long_form, description, param_type=str):
        self.has_optional = True
        self.optional_arguments[short_form] = dict()
        self.optional_arguments[short_form]['sh'] = short_form
        self.optional_arguments[short_form]['lf'] = long_form
        self.optional_arguments[short_form]['des'] = description
        self.optional_arguments[short_form]['type'] = param_type

    def add_compulsory_arguments(self, short_form, long_form, description, param_type=str):
        self.has_compulsory = True
        self.compulsory_arguments[short_form] = dict()
        self.compulsory_arguments[short_form]['sh'] = short_form
        self.compulsory_arguments[short_form]['lf'] = long_form
        self.compulsory_arguments[short_form]['des'] = description
        self.compulsory_arguments[short_form]['type'] = param_type

    def decode_argument(self, options, vals, is_compulsory=False):
        res = dict()
        for v in vals:
            try:
                pos = options.index(v['sh'])
                if options.count(v['sh']) > 1 or options.count(v['lf']) > 0:
                    print('Duplicate options found for ' + v['sh'])
                    return None, None
                remove_text = 'sh'
            except ValueError:
                try:
                    pos = options.index(v['lf'])
                    if options.count(v['lf']) > 1 or options.count(v['sh']) > 0:
                        print('Duplicate options found' + v['sh'])
                        return None, None
                    remove_text = 'lf'
                except ValueError:
                    if is_compulsory:
                        print(v['sh'] + " or " + v['lf'] + " not present in the options")
                        return None, None
                    else:
                        continue
            if v['type'] is None:
                res[v['sh']] = None
                options.remove(v[remove_text])
                continue
            try:
                if options[pos + 1][0] == '-':
                    raise IndexError
                else:
                    if v['type'] == bool:
                        if options[pos + 1] == 'true':
                            val = True
                        elif options[pos + 1] == 'false':
                            val = False
                        else:
                            raise ValueError
                    else:
                        val = v['type'](options[pos + 1])
                    res[v['sh']] = val
                    options.remove(v[remove_text])
                    options.remove(options[pos])
            except IndexError:
                print("No value is given for option " + v['sh'])
                return None, None
            except ValueError:
                print("Wrong value is given for option " + v['sh'])
                return None, None
        return res, options

    def decode_options(self, options):

        res, options = self.decode_argument(options, self.optional_arguments.values())
        if res is None:
            return None

        if '-h' in list(res.keys()):
            return res

        res_comp, options = self.decode_argument(options, self.compulsory_arguments.values(), is_compulsory=True)
        if res_comp is None:
            return None
        res.update(res_comp)

        if self.has_positional:
            if len(options) < len(self.positional_arguments):
                print("All positional arguments are not found")
                return None

            for k, v in self.positional_arguments.items():
                try:
                    if v['type'] == bool:
                        if options[0] == 'true':
                            val = True
                        elif options[0] == 'false':
                            val = False
                        else:
                            raise ValueError
                    else:
                        val = v['type'](options[0])
                    res[v['sh']] = val
                except ValueError:
                    print("Wrong value is given for the position "+str(k))
                    return None

        if self.inf_positional:
            i = 1
            for v in options:
                res['inf'+str(i)] = v
                i += 1
        else:
            if len(options) > 1:
                print("some arguments, not required, have been ignored")
        return res


class StrArgParser:

    def write_file(self, line, end="\n"):
        self.f_tmp.write(str(line)+end)

    def __init__(self, description=""):
        self.commands = dict()
        self.f_tmp = None
        self.description = description

        self.add_command('ls_cmd', 'Lists all the available command with usage', function=self.cmd_ls_cmd)
        self.get_command('ls_cmd').add_optional_arguments('-v', '--verbose', "Give the output in detail",
                                                          param_type=None)

    def __repr__(self):
        return self.description

    def get_command(self, name):
        return self.commands[name]

    def add_command(self, command, description, inf_positional=False, function=None):
        c = Command(command, description, inf_positional, function)
        c.add_optional_arguments('-h', '--help', 'Gives the details of the command', param_type=None)
        c.add_optional_arguments('->', '->_route', 'Overwrite the output to the file')
        c.add_optional_arguments('->>', '->>_route', 'Append the output to the file')
        self.commands[command] = c

    def cmd_ls_cmd(self, res, out_func=print):
        is_verbose = len(res) > 0
        for k, v in self.commands.items():
            out_func("Command: " + k),
            if is_verbose:
                out_func(v)
                out_func("\n"+v.description+"\n\n\t\t---x---\n")

    def show_help(self, res, out_func=print):
        for k, v in self.commands.items():
            out_func("Command "+k)
            v.show_help(out_func=out_func)
            out_func("\t\t----x----\n")

    def decode_command(self, s):
        s = s.strip(' ')
        s = s.split(' ')
        try:
            s.remove('')
        except ValueError:
            pass
        try:
            res = self.commands[s[0]].decode_options(s[1:])
            out_func = print

            if res is None:
                return None, None, None, print
            ls_key = list(res.keys())
            if '->' in ls_key or '->>' in ls_key:
                if '->' in ls_key:
                    self.f_tmp = open(res['->'], 'w')
                else:
                    self.f_tmp = open(res['->>'], 'a')
                out_func = self.write_file
            if '-h' in ls_key:
                self.commands[s[0]].show_help(out_func=out_func)
                if self.f_tmp is not None:
                    self.f_tmp.close()
                    self.f_tmp = None
                return None, None, None, None

            return s[0], res, self.commands[s[0]].function, out_func
        except KeyError:
            print("Command not found. Use help command for details on various commands")
            return None, None, None, print
