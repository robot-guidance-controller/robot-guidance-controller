import os
import threading
import queue
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.connection import Listener
import math

command_queue = queue.Queue()
client_windows = {}


class LineEntry:
    def __init__(self, options=None):
        self.options = options or {}
        self.data = None
        self.x_data = None
        self.y_data = None

    def update_data(self, new_data, mode="append"):
        if mode == "replace":
            self.data = None
            self.x_data = None
            self.y_data = None

        if isinstance(new_data, np.ndarray):
            new_data = new_data.tolist()

        if isinstance(new_data, tuple):
            new_data = tuple(x.tolist() if isinstance(
                x, np.ndarray) else x for x in new_data)
            if len(new_data) == 2:
                if all(isinstance(d, (list, tuple)) for d in new_data):
                    self.x_data = list(new_data[0])
                    self.y_data = list(new_data[1])
                    self.data = None
                elif all(isinstance(d, (int, float)) for d in new_data):
                    if self.data is None:
                        self.data = []
                    self.data.append(new_data)
                else:
                    if self.data is None:
                        self.data = []
                    self.data.append(new_data)
            else:
                if self.data is None:
                    self.data = []
                self.data.append(new_data)
        elif isinstance(new_data, list):
            new_data = [x.tolist() if isinstance(x, np.ndarray)
                        else x for x in new_data]
            if new_data and all(isinstance(item, (list, tuple)) and len(item) == 2 and
                                all(isinstance(val, (int, float))
                                    for val in item)
                                for item in new_data):
                xs, ys = zip(*new_data)
                self.x_data = list(xs)
                self.y_data = list(ys)
                self.data = None
            else:
                self.data = list(new_data)
                self.x_data = None
                self.y_data = None
        else:
            if self.data is None:
                self.data = []
            self.data.append(new_data)

    def update_config(self, options):
        self.options.update(options)

    def get_data(self):
        if self.x_data is not None and self.y_data is not None:
            return np.array(self.x_data), np.array(self.y_data)
        elif self.data is not None:
            if len(self.data) > 0 and isinstance(self.data[0], (list, tuple)):
                data_arr = np.array(self.data)
                if data_arr.ndim == 2 and data_arr.shape[1] >= 2:
                    return data_arr[:, 0], data_arr[:, 1]
                else:
                    return np.arange(len(self.data)), data_arr
            else:
                return np.arange(len(self.data)), np.array(self.data)
        else:
            return None, None


class PlotEntry:
    def __init__(self, options=None):
        self.options = options or {}
        self.lines = {}
        self.create_line("default", {})

    def create_line(self, line_id, options):
        if line_id not in self.lines:
            self.lines[line_id] = LineEntry(options)
        else:
            self.lines[line_id].update_config(options)

    def update_line(self, line_id, data, mode="append"):
        if line_id not in self.lines:
            self.lines[line_id] = LineEntry()
        self.lines[line_id].update_data(data, mode)

    def config_line(self, line_id, options):
        if line_id in self.lines:
            self.lines[line_id].update_config(options)

    def remove_line(self, line_id):
        if line_id in self.lines:
            del self.lines[line_id]

    def update_config(self, options):
        self.options.update(options)

    def draw(self, ax):
        for line_id, line in sorted(self.lines.items()):
            x, y = line.get_data()
            if x is not None and y is not None:
                opts = line.options
                plot_style = opts.get("plot_style", "-o")
                markersize = opts.get("markersize", 4)
                label = opts.get("label", line_id)
                color = opts.get("color", None)
                ax.plot(x, y, plot_style, markersize=markersize,
                        label=label, color=color)
        if self.lines:
            ax.legend()


class ClientWindow:
    def __init__(self, client_id):
        self.client_id = client_id
        # Mapping: plot_id -> PlotEntry
        self.plots = {}
        self.lock = threading.Lock()
        self.fig = plt.figure()
        self.fig.suptitle(f"Client {client_id} Plots")

    def create_plot(self, plot_id, options):
        with self.lock:
            if plot_id not in self.plots:
                self.plots[plot_id] = PlotEntry(options)
            else:
                self.plots[plot_id].update_config(options)

    def update_plot(self, plot_id, data, mode="append"):
        with self.lock:
            if plot_id not in self.plots:
                self.plots[plot_id] = PlotEntry()
            self.plots[plot_id].update_line("default", data, mode)

    def config_plot(self, plot_id, options):
        with self.lock:
            if plot_id in self.plots:
                self.plots[plot_id].update_config(options)

    def remove_plot(self, plot_id):
        with self.lock:
            if plot_id in self.plots:
                del self.plots[plot_id]

    def create_line(self, plot_id, line_id, options):
        with self.lock:
            if plot_id not in self.plots:
                self.plots[plot_id] = PlotEntry()
            self.plots[plot_id].create_line(line_id, options)

    def update_line(self, plot_id, line_id, data, mode="append"):
        with self.lock:
            if plot_id not in self.plots:
                self.plots[plot_id] = PlotEntry()
            self.plots[plot_id].update_line(line_id, data, mode)

    def config_line(self, plot_id, line_id, options):
        with self.lock:
            if plot_id in self.plots:
                self.plots[plot_id].config_line(line_id, options)

    def remove_line(self, plot_id, line_id):
        with self.lock:
            if plot_id in self.plots:
                self.plots[plot_id].remove_line(line_id)

    def refresh(self):
        with self.lock:
            num_plots = len(self.plots)
            if num_plots == 0:
                print("No plots available.")
                return
            self.fig.clf()
            # Calculate grid dimensions
            cols = int(math.ceil(math.sqrt(num_plots)))
            rows = int(math.ceil(num_plots / cols))
            axes = self.fig.subplots(rows, cols)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.flatten()
            sorted_plots = sorted(self.plots.items())
            for i, ax in enumerate(axes):
                if i < num_plots:
                    ax.clear()
                    plot_id, plot_entry = sorted_plots[i]
                    opts = plot_entry.options
                    ax.set_title(opts.get("title", f"Plot {plot_id}"))
                    ax.set_xlabel(opts.get("xlabel", "X"))
                    ax.set_ylabel(opts.get("ylabel", "Y"))
                    ax.grid(visible=True, alpha=0.25)
                    if "xlim" in opts:
                        ax.set_xlim(opts["xlim"])
                    if "ylim" in opts:
                        ax.set_ylim(opts["ylim"])
                    plot_entry.draw(ax)
                else:
                    ax.set_visible(False)
            self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


class ClientHandler(threading.Thread):
    def __init__(self, conn, client_id):
        super().__init__(daemon=True)
        self.conn = conn
        self.client_id = client_id

    def run(self):
        # On connection, register the client.
        command_queue.put(('register_client', self.client_id))
        try:
            while True:
                msg = self.conn.recv()  # Expect a dict message.
                command_queue.put((self.client_id, msg))
        except EOFError:
            command_queue.put(('remove_client', self.client_id))
        finally:
            self.conn.close()
            print(f"Client {self.client_id} disconnected.")


class PlotService:
    def __init__(self, address=None, authkey=b'secret password'):
        if address is None:
            address = r'\\.\pipe\plot_service' if os.name == 'nt' else '/tmp/plot_service.sock'
        self.address = address
        self.authkey = authkey
        if os.name != 'nt' and os.path.exists(address):
            os.unlink(address)
        self.listener = Listener(address, authkey=authkey)
        self.client_id_counter = 0
        self.handlers = []

    def start(self):
        print(f"Plot service started at {self.address}")
        accept_thread = threading.Thread(
            target=self.accept_clients, daemon=True)
        accept_thread.start()
        self.run_plot_loop()

    def accept_clients(self):
        while True:
            try:
                conn = self.listener.accept()
            except Exception as e:
                print("Listener error:", e)
                break
            self.client_id_counter += 1
            handler = ClientHandler(conn, self.client_id_counter)
            handler.start()
            self.handlers.append(handler)
            print(f"Accepted client {self.client_id_counter}")

    def run_plot_loop(self):
        try:
            while True:
                while not command_queue.empty():
                    item = command_queue.get()
                    if isinstance(item, tuple) and item[0] == 'register_client':
                        _, client_id = item
                        if client_id not in client_windows:
                            client_windows[client_id] = ClientWindow(client_id)
                    elif isinstance(item, tuple) and item[0] == 'remove_client':
                        _, client_id = item
                        if client_id in client_windows:
                            print(f"Removing all plots for client {client_id}")
                            client_windows[client_id].close()
                            del client_windows[client_id]
                    else:
                        client_id, msg = item
                        action = msg.get("action")
                        plot_id = msg.get("plot_id")
                        if client_id not in client_windows:
                            client_windows[client_id] = ClientWindow(client_id)
                        cw = client_windows[client_id]
                        if action == "create":
                            options = msg.get("options", {})
                            cw.create_plot(plot_id, options)
                        elif action == "update":
                            data = msg.get("data")
                            mode = msg.get("mode", "append")
                            cw.update_plot(plot_id, data, mode)
                        elif action == "config":
                            options = msg.get("options", {})
                            cw.config_plot(plot_id, options)
                        elif action == "remove":
                            cw.remove_plot(plot_id)
                        elif action == "create_line":
                            line_id = msg.get("line_id", "default")
                            options = msg.get("options", {})
                            cw.create_line(plot_id, line_id, options)
                        elif action == "update_line":
                            line_id = msg.get("line_id", "default")
                            data = msg.get("data")
                            mode = msg.get("mode", "append")
                            cw.update_line(plot_id, line_id, data, mode)
                        elif action == "config_line":
                            line_id = msg.get("line_id", "default")
                            options = msg.get("options", {})
                            cw.config_line(plot_id, line_id, options)
                        elif action == "remove_line":
                            line_id = msg.get("line_id", "default")
                            cw.remove_line(plot_id, line_id)
                for cw in list(client_windows.values()):
                    cw.refresh()
        except KeyboardInterrupt:
            print("Plot service shutting down.")
        finally:
            self.listener.close()
            plt.ioff()
            plt.show()


if __name__ == '__main__':
    plt.ion()
    service = PlotService()
    service.start()
