import subprocess

class Vocalizer:
    def __init__(self):
        self._last_proc = None

    def is_uttering(self) -> bool:
        return self._last_proc is not None and self._last_proc.poll() is None

    def utter(self, phrase: str, interupt: bool = False) -> bool:
        if self.is_uttering():
            if not interupt:
                return False
            self._last_proc.terminate()

        self._last_proc = subprocess.Popen(["say", phrase])
        
        return True

    def close(self):
        if self._last_proc is not None:
            self._last_proc.terminate()
            self._last_proc = None