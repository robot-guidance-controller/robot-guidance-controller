import threading
from typing import Dict, Optional, Set
from stoppable_thread import StoppableThread

_exit_command = 'exit'

def _console_command_repl(stop_event: threading.Event, command_events: Dict[str, threading.Event]):
    possible_commands_ouput = 'Possible commands: ' + ', '.join(f"'{command}'" for command in command_events.keys()) + '.'
    output = possible_commands_ouput

    while not stop_event.is_set():
        command = input(output)

        output = possible_commands_ouput

        if command not in command_events:
            output = f"Invalid command: '{command}'.\n" + output
            continue
        
        command_events[command].set()

        if command == _exit_command:
            break

class ConsoleCommandThread(StoppableThread):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConsoleCommandThread, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, commands: Optional[Set[str]] = None):
        if hasattr(self, 'initialized'):
            return

        self.initialized = True

        if commands is None:
            commands = []

        if _exit_command not in commands:
            commands.append(_exit_command)

        self.command_events = {}

        for command in commands:
            if not isinstance(command, str):
                raise ValueError(f"Invalid command: '{command}'. Command must be a string.")
            self.command_events[command] = threading.Event()

        super().__init__(
            stoppable_method=_console_command_repl,
            stoppable_method_args=self.command_events,
            name='console_command_thread'
        )

    def stop(self):
        super().stop()
        self.__class__._instance = None
        if hasattr(self, 'initialized'):
            del self.initialized

    def poll_command(self, command: str) -> bool:
        command_event_status = self.command_events[command].is_set()
        self.command_events[command].clear()
        return command_event_status
    
    def reset(self):
        for command_event in self.command_events.values():
            command_event.clear()