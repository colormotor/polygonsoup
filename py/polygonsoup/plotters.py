'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

plotters - plotter/drawing machine interfaces
'''

import socket, sys
import numpy as np
import time
import polygonsoup.geom as geom
from polygonsoup.plut import NoPlotter
# class NoPlotter:
#     '''Default dummy plotter
#       Use AxiDrawClient or AxiPlotter to plot somethign
#     '''

#     def __init__(self):
#         pass

#     def _set_bounds(self, w, h):
#         pass

#     def _stroke(self, P):
#         pass

#     def _plot(self, title='', padding=0, box=None):
#         pass

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

    def _plot(self, title='', padding=0, box=None):
        try:
            import axi
            if box is None:
                box = geom.bounding_box(self.paths)
            dstbox = geom.make_rect(0, 0, *self.bounds)
            mat = geom.rect_in_rect_transform(box, geom.make_rect(0, 0, *self.bounds), padding)
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
    def __init__(self, address_or_settings='./client_settings.json', port=None, raw=False): #, blocking=False):
        """
        :param address_or_settings:  (Default value = './client_settings.json') This parameter can specify either an IP address for a server,
        or the path to a json file containing the connection settings. The json file must contain the entries 'address' and 'port' with
        the IP address and the port number of the server.
        :param port:  (Default value = None) This parameter is required if adress_or_settings explicitly defines an IP address.
        If a json settings file is specified, defining this parameter will override the port defined in the file.
        :param raw:  (Default value = False): This will send coordinates (unscaled) to the server that will not be automatically scaled.
        Avoid using unless the coordinate system of the drawing fits in the drawing area.
        :param blocking:  (Default value = False)
        """
        if '.json' in address_or_settings:
            import json
            try:
                settings = json.loads(open(address_or_settings).read())
                self.address = settings['address']
                self.port = settings['port']
            except FileNotFoundError as e:
                print(e)
                self.address = None
            if port is not None:
                self.port = port
        else:
            self.address = address_or_settings
            self.port = port
        self.socket_open = False
        self.sock = None
        self.paths = []
        self.bounds = None
        self.raw = raw
        self.print_err = True

    def open(self):
        if self.address == None:
            return

        server_address = (self.address, self.port)

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #print('connecting to %s port %s'%server_address)
            self.sock.connect(server_address)
            self.socket_open = True
        except ConnectionRefusedError as e:
            if self.print_err:
                print(e)
                print('could not connect to: ' + str(server_address))
                self.print_err = False
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
        if not len(P):
            return
        if len(P[0])==2:
            self.sendln('PATHCMD stroke %d %s'%path_to_str(P))
        elif len(P[0])==3:
            self.sendln('PATHCMD stroke3 %d %s'%path_to_str(P))
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

    def _plot(self, title='', padding=0, box=None):
        if self.raw:
            print('Resizing raw plot')
            if box is None:
                box = geom.bounding_box(self.paths)
            mat = geom.rect_in_rect_transform(box, geom.make_rect(0, 0, *self.bounds), padding)
            # Assume we might have a z coordinate here
            self.paths = [np.array(P) for P in self.paths]
            for i in range(len(self.paths)):
                self.paths[i][:,:2] = geom.affine_transform(mat, self.paths[i][:,:2])
            #self.paths = geom.affine_transform(mat, self.paths)
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
    if len(P[0]) == 2:
        return len(P), ' '.join(['%f %f'%(p[0], p[1]) for p in P])
    return len(P), ' '.join(['%f %f %f'%(p[0], p[1], p[2]) for p in P])


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
