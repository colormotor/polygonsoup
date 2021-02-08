#!/usr/bin/env python3
import socket, sys
import numpy as np
import time
import polygonsoup.geom as geom

class NoPlotter:
    '''Default dummy plotter
      Use AxiDrawClient or AxiPlotter to plot somethign
    '''

    def __init__(self):
        pass

    def _set_bounds(self, w, h):
        pass

    def _stroke(self, P):
        pass

    def _plot(self, title='', padding=0):
        pass

class AxiPlotter:
    ''' Direct connection to axi module'''
    def __init__(self, sort=False):
        self.paths = []
        self.sort = sort

    # Interface with plot module
    def _set_bounds(self, w, h):
        self.bounds = (w, h)
        pass

    def _stroke(self, P):
        self.paths.append(P)

    def _plot(self, title='', padding=0):
        try:
            import axi
            srcbox = geom.bounding_box(self.paths)
            dstbox = geom.make_rect(0, 0, *self.bounds)
            mat = geom.rect_in_rect_transform(srcbox, geom.make_rect(0, 0, *self.bounds), padding)
            self.paths = geom.affine_transform(mat, self.paths)
            paths = [[tuple(p) for p in path] for path in self.paths]
            if self.sort and len(paths) > 1:
                paths = axi.sort_paths(paths)
            d = axi.Drawing(paths)

            if title:
                text_pos = (geom.rect_l(dstbox), geom.rect_b(dstbox))
                font = axi.Font(axi.FUTURAL, 7.5) #
                dtext = font.text(title)
                dtext = dtext.translate(*text_pos)
                d.add(dtext)

            try:
                axi.draw(d)
            except Exception as e:
                print(e)

        except ModuleNotFoundError as e:
            print(e)
            print('Could not find axi module')

class AxiDrawClient:
    ''' Plots to a remote instance of axidraw_server.py'''
    def __init__(self, address_or_settings='./client_settings.json', port=80, raw=False): #, blocking=False):
        if '.json' in address_or_settings:
            import json
            settings = json.loads(open(address_or_settings).read())
            self.address = settings['address']
            self.port = settings['port']
        else:
            self.address = address_or_settings
            self.port = port
        self.socket_open = False
        self.sock = None
        self.paths = []
        self.bounds = None
        self.raw = raw

    def open(self):
        server_address = (self.address, self.port)

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print('connecting to %s port %s'%server_address)
            self.sock.connect(server_address)
            self.socket_open = True
        except ConnectionRefusedError as e:
            print(e)
            print('could not connect to: ' + str(server_address))
            self.sock = None
            self.socket_open = False

    def close(self):
        print('Closing socket')
        if self.sock is not None:
            self.sock.close()
            self.socket_open = False
            self.paths.clear()

    def send(self, msg):
        auto_open = False
        if not self.socket_open:
            self.open()
            auto_open = True
        if self.socket_open:
            self.sock.sendall(msg.encode('utf-8'))
            if auto_open:
                self.close()

    def sendln(self, msg):
        self.send(msg + '\n')

    def drawing_start(self, title=''):
        self.open()
        if title:
            self.sendln('PATHCMD title ' + title)
        self.sendln('PATHCMD drawing_start')

    def drawing_end(self, close=False):
        if self.raw:
            self.drawing_end_raw()
            return

        self.sendln('PATHCMD drawing_end')
        if close:
            self.close()

    def drawing_end_raw(self):
        self.sendln('PATHCMD drawing_end_raw')
        self.close()

    def draw_paths(self, S, title='', close=False):
        try:
            self.drawing_start(title)
            for P in S:
                self.add_path(P)
            self.drawing_end(close)

        except ConnectionRefusedError as e:
            print('could not connect to network')
            print(e)

    def drawing(self, drawing, title=''):
        try:
            self.drawing_start(title)
            for P in drawing.paths:
                self.add_path(P)
            self.drawing_end()
        except ConnectionRefusedError as e:
            print('could not connect to network')
            print(e)

    def wait(self):
        print('waiting')
        self.sendln('wait')
        rep = recv_line(self.sock)
        if rep == 'done':
            print('Finished waiting')
            return True
        return False

    def add_path(self, P):
        self.sendln('PATHCMD stroke %d %s'%path_to_str(P))

    def pen_up(self):
        self.sendln('PATHCMD pen_up')

    def pen_down(self):
        self.sendln('PATHCMD pen_down')

    def home(self):
        self.sendln('PATHCMD home')

    # Interface with plot module
    def _set_bounds(self, w, h):
        self.bounds = (w, h)
        pass

    def _stroke(self, P):
        self.paths.append(P)

    def _plot(self, title='', padding=0):
        if self.raw:
            print('Resizing raw plot')
            srcbox = geom.bounding_box(self.paths)
            mat = geom.rect_in_rect_transform(srcbox, geom.make_rect(0, 0, *self.bounds), padding)
            self.paths = geom.affine_transform(mat, self.paths)
        self.draw_paths(self.paths, title, close=True)

    # Visualization (use plot.py instead)
    def visualize_drawing(self, drawing, title='', close=False, figsize=(7,7), axis=False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        if title:
            plt.title(title)
        for P in drawing.paths:
            plt.plot([p[0] for p in P],
                     [p[1] for p in P], 'k', linewidth=0.5)
        plt.axis('equal')
        if not axis:
            plt.axis('off')
        plt.gca().invert_yaxis()
        plt.show()

    def visualize_paths(self, S, title='', close=False, figsize=(7,7), axis=False):
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        if title:
            plt.title(title)
        if type(S) != list:
            S = [S]
        for P in S:
            if type(P) == np.ndarray:
                P = list(P.T)
            if close:
                P = P + [P[0]]
            plt.plot([p[0] for p in P],
                     [p[1] for p in P], 'k', linewidth=0.5)
        plt.axis('equal')
        if not axis:
            plt.axis('off')
        plt.gca().invert_yaxis()
        plt.show()


def path_to_str(P):
    ''' Convert a path to a (num_points, point sequence) tuple'''
    #if type(P) == np.ndarray:
    #    P = P.T
    return len(P), ' '.join(['%f %f'%(p[0], p[1]) for p in P])

def recv_line(sock):
    s = ''
    while True:
        off = s.find("\n")
        if -1 != off: break
        #print("reading more from connection")
        buf = sock.recv(1024)
        if not buf: return ''
        if buf[0] == 0xff: return ''
        if buf == '': return ''
        buf = buf.decode("utf-8")
        s += buf
    ret = s[0:off]
    ret = ret.rstrip()
    print('received ' + ret)
    return ret
