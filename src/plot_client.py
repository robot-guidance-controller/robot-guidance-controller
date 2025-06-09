from multiprocessing.connection import Client
import os


class PlotClient:
    def __init__(self, address=None, authkey=b'secret password'):
        if address is None:
            address = r'\\.\pipe\plot_service' if os.name == 'nt' else '/tmp/plot_service.sock'
        self.conn = Client(address, authkey=authkey)

    def create_plot(self, plot_id, **options):
        msg = {
            "action": "create",
            "plot_id": plot_id,
            "options": options,
        }
        self.conn.send(msg)

    def update_plot(self, plot_id, data, mode="append"):
        msg = {
            "action": "update",
            "plot_id": plot_id,
            "data": data,
            "mode": mode,
        }
        self.conn.send(msg)

    def config_plot(self, plot_id, **options):
        msg = {
            "action": "config",
            "plot_id": plot_id,
            "options": options,
        }
        self.conn.send(msg)

    def remove_plot(self, plot_id):
        msg = {
            "action": "remove",
            "plot_id": plot_id,
        }
        self.conn.send(msg)

    def create_line(self, plot_id, line_id, **options):
        msg = {
            "action": "create_line",
            "plot_id": plot_id,
            "line_id": line_id,
            "options": options,
        }
        self.conn.send(msg)

    def update_line(self, plot_id, line_id, data, mode="append"):
        msg = {
            "action": "update_line",
            "plot_id": plot_id,
            "line_id": line_id,
            "data": data,
            "mode": mode,
        }
        self.conn.send(msg)

    def config_line(self, plot_id, line_id, **options):
        msg = {
            "action": "config_line",
            "plot_id": plot_id,
            "line_id": line_id,
            "options": options,
        }
        self.conn.send(msg)

    def remove_line(self, plot_id, line_id):
        msg = {
            "action": "remove_line",
            "plot_id": plot_id,
            "line_id": line_id,
        }
        self.conn.send(msg)

    def close(self):
        self.conn.close()
