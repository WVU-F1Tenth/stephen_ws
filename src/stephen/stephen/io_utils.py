from dataclasses import dataclass, fields
from typing import Any, Union, Dict, List
import sys
import select
import tty
import termios
from collections import defaultdict

@dataclass
class Binding:
    name: str
    key: str
    v: Union[bool, float]

@dataclass
class DualBinding:
    name: str
    on_key: str
    off_key: str
    v: bool

class KeyBindings:
    
    def __init__(self, **bindings):
        """
        Takes dataclass of bindings.
        Each attribute is a single binding.
        Each binding defines: v, key, and name.
        """
        bindings['speed']  = Binding('speed', 's', 0.0)
        self.params: Dict[str, Union[Binding, DualBinding]] = bindings
        self.key_dict = defaultdict(list)
        for binding in bindings.values():
            if isinstance(binding, Binding):
                if binding.key:
                    self.key_dict[binding.key].append(binding)
            elif isinstance(binding, DualBinding):
                if binding.on_key:
                    self.key_dict[binding.on_key].append(binding)
                if binding.off_key:
                    self.key_dict[binding.off_key].append(binding)
            else:
                raise ValueError('Invalid type passed to KeyBindings')

        if bindings.get('speed'):
            self.selected = bindings['speed']
        else:
            raise ValueError('speed keybinding must be defined')

        # Keybinding message
        key_message = '\n'.join(
            [f'  {param.key}={param.name}' for param in self.params.values() if isinstance(param, Binding)])
        self.message = f'Commands:\n  <space>=stop\n{key_message}'
        print(self.message)

        # Terminal handling
        self.fd = sys.stdin.fileno()
        self.terminal_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        
    def add_state(self, name: str, v: bool):
        self.__dict__[name] = v

    def check_input(self):
        key = self.get_key()
        if not key:
            return
        elif key == '\t':
            print(self.message)
        elif key == ' ':
            self.speed.v = 0.0 # type: ignore
            self.stopped = True
            print('stopped')
        elif key in ('i', 'o', 'j', 'k', 'n', 'm'):
            if key == 'i':
                self.selected.v -= 1.0
            elif key == 'o':
                self.selected.v += 1.0
            elif key == 'j':
                self.selected.v -= 0.1
            elif key == 'k':
                self.selected.v += 0.1
            elif key == 'n':
                self.selected.v -= 0.01
            elif key == 'm':
                self.selected.v += 0.01
            print(f'{self.selected.name} = {self.selected.v:.2f}')
        else:
            binding_list = self.key_dict.get(key)
            if not binding_list:
                return
            for binding in binding_list:
                if isinstance(binding, Binding) and isinstance(binding.v, float):
                    self.selected = binding
                    print(f'{binding.name} selected')
                elif isinstance(binding, Binding) and isinstance(binding.v, bool):
                    if binding.v:
                        binding.v = False
                    else:
                        binding.v = True
                        print(binding.name)
                elif isinstance(binding, DualBinding):
                    if binding.on_key == key:
                        binding.v = True
                    else:
                        binding.v = False

    def __getattr__(self, name: str) -> Union[Binding, DualBinding]:
        try:
            return self.params[name]
        except KeyError:
            raise AttributeError(f"{name} key binding doesn't exist")
        
    def get_key(self):
        rlist, _, _ = select.select([sys.stdin], [], [], 0.005)
        if rlist:
            return sys.stdin.read(1)
        return None
    
    def restore_terminal(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.terminal_settings)

class FileInfo:
    pass

class PrintInfo:
    pass