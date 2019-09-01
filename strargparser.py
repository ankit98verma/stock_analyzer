

class Command:

    def __init__(self, command_name, description):
        self.description = description
        self.command_name = command_name
        self.positional_arguments = dict()
        self.compulsory_arguments = dict()
        self.optional_arguments = dict()

        self.add_optional_arguments('Help', '-h', '--help', 'Gives the details of the command')

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

        return string

    def show_help(self):
        string = self.__repr__()
        string += "\n\n"
        string += self.description + "\n"
        string += "\n"
        if self.has_positional:
            string += "positional arguments (all compulsory):\n"
            for v in self.positional_arguments.values():
                string += "\t"+v['sh']+"\t"+v['lf']+"\t"+v['des']+"\n"

        if self.has_compulsory:
            string += "compulsory arguments with options:\n"
            for v in self.compulsory_arguments.values():
                string += "\t"+v['sh']+"\t"+v['lf']+"\t"+v['des']+"\n"

        if self.has_optional:
            string += "optional arguments with options:\n"
            for v in self.optional_arguments.values():
                string += "\t" + v['sh'] + "\t" + v['lf'] + "\t" + v['des'] + "\n"

        print(string)

    def add_positional_arguments(self, position, short_form, long_form, description):
        self.has_positional = True
        self.positional_arguments[position] = dict()
        self.positional_arguments[position]['sh'] = short_form
        self.positional_arguments[position]['lf'] = long_form
        self.positional_arguments[position]['des'] = description

    def add_optional_arguments(self, name, short_form, long_form, description):
        self.has_optional = True
        self.optional_arguments[name] = dict()
        self.optional_arguments[name]['sh'] = short_form
        self.optional_arguments[name]['lf'] = long_form
        self.optional_arguments[name]['des'] = description

    def add_compulsory_arguments(self, name, short_form, long_form, description):
        self.has_compulsory = True
        self.compulsory_arguments[name] = dict()
        self.compulsory_arguments[name]['sh'] = short_form
        self.compulsory_arguments[name]['lf'] = long_form
        self.compulsory_arguments[name]['des'] = description

    def get_short_list(self, get_dict):
        res = []
        for v in get_dict.values():
            res.append(v['sh'])
        return res

    def get_long_list(self, get_dict):
        res = []
        for v in get_dict.values():
            res.append(v['lf'])
        return res

    def decode_options(self, options):
        res = dict()
        if '-h' in options or '--help' in options:
            self.show_help()
            return None

        if self.has_compulsory:
            for v in self.compulsory_arguments.values():
                try:
                    pos = options.index(v['sh'])
                    if options.count(v['sh']) > 1 or options.count(v['lf']) > 0:
                        print('Duplicate options found for ' + v['sh'])
                        return None
                    remove_text = 'sh'
                except ValueError:
                    try:
                        pos = options.index(v['lf'])
                        if options.count(v['lf']) > 1 or options.count(v['sh']) > 0:
                            print('Duplicate options found' + v['sh'])
                            return None
                        remove_text = 'lf'
                    except ValueError:
                        print(v['sh']+" or "+v['lf'] + " not present in the options")
                        return None
                if pos != -1:
                    try:
                        if options[pos+1][0] == '-':
                            raise IndexError
                        else:
                            res[v['sh']] = options[pos + 1]
                            options.remove(v[remove_text])
                            options.remove(options[pos])
                    except IndexError:
                        print("No value given for option "+v['sh'])
                        return None

        if not self.has_optional and not self.has_positional and len(options):
            print("some options, not required, have been ignored")
            return res

        optional_short_list = self.get_short_list(self.optional_arguments)
        optional_long_list = self.get_long_list(self.optional_arguments)
        while self.has_optional and len(options):
            start = -1
            for k in options:
                if len(k) > 0:
                    if k[0] == '-':
                        start = options.index(k)
                        break
                else:
                    break
            if start == -1:
                break

            if options.count(options[start]) > 1:
                print("Duplicate options found"+options[start])
                return None

            if options[start] in optional_short_list:
                res_key = options[start]
            elif options[start] in optional_long_list:
                index = optional_long_list.index(options[start])
                res_key = optional_short_list[index]
            else:
                print("Unknown argument "+options[start])
                return None

            try:
                if options[start+1][0] == '-':
                    raise IndexError
                else:
                    res[res_key] = options[start+1]
                    options.remove(options[start])
                    options.remove(options[start])
            except IndexError:
                print("No value given for option "+options[start])
                return None

        if self.has_positional:
            if len(options) != len(self.positional_arguments):
                print("All positional arguments are not found")
                return None
            i = 0
            for k, v in self.positional_arguments.items():
                res[k] = options[0]
                i += 1
            print("Taking care of positional arguments")

        if len(options) > 1:
            print("some arguments, not required, have been ignored")
        return res


class StrArgParser:

    def __init__(self, description=""):
        self.commands = dict()
        self.description = description

    def __repr__(self):
        return self.description

    def __getattr__(self, name):
        if name in self.commands:
            return self.commands[name]
        else:
            return self.__getattribute__(name)

    def add_command(self, command, description):
        c = Command(command, description)
        self.commands[command] = c

    def show_help(self):
        for k, v in self.commands.items():
            print("Command "+k)
            v.show_help()
            print("\t\t----x----\n")

    def decode_command(self, s):
        s = s.strip(' ')
        s = s.split(' ')
        try:
            s.remove('')
        except ValueError:
            pass
        try:
            return s[0], self.commands[s[0]].decode_options(s[1:])
        except KeyError:
            print("Command not found. Use help command for details on various commands")
            return None
