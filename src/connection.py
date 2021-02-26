import pickle
import socket


class Connection(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = self.sock
        self.client = True
        self.open = False

    def __del__(self):
        self.stop()

    def recvall(self):
        data = self.conn.recv(4096)
        n = pickle.loads(data)
        data = data[len(pickle.dumps(n)):]

        while n != len(data):
            data += self.conn.recv(4096)

        data = pickle.loads(data)
        return data

    def sendall(self, msg):
        data = pickle.dumps(msg)
        n = len(data)
        data = pickle.dumps(n) + data
        self.conn.sendall(data)
        return

    def init_client(self, server: dict, dataset: str, topk: int, langs: list, search_params: dict):
        print("Initializing Client")
        self.sock.connect((server['host'], server['port']))
        msg = {'init': dict(dataset=dataset, n=topk, langs=langs, b=search_params['b'], k1=search_params['k1'])}
        for k, v in msg['init'].items():
            print("{0: <12}: {1}".format(k, v))
        self.sendall(msg)
        self.open = True

    def init_server(self, port):
        self.sock.bind((socket.gethostname(), port))
        self.sock.listen()
        self.open = True

    def accept(self):
        (conn, addr) = self.sock.accept()
        print("Connected to {}".format(addr))
        self.conn = conn

    def close(self, stop=False):
        print("Stopping")
        if not self.open:
            return
        msg = {'stop': stop}
        self.sendall(msg)
        try:
            self.conn.close()
        except OSError:
            pass

    def stop(self):
        """ Stopping server"""
        try:
            self.sock.close()
        except OSError:
            pass

    def search(self, question, lang, field):
        msg = {'search': [{'lang': lang, 'question': question, 'id': 0, 'field': field}]}
        self.sendall(msg)
