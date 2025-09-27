'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

plotters - plotter/drawing machine interfaces
'''

import socket, sys, copy
import numpy as np
import time, os
import polygonsoup.geom as geom

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

def sort_paths(S):
    import axi
    S = [[tuple(p) for p in path] for path in paths]
    sorted_paths = axi.sort_paths(S, reversable = True)
    sorted_paths = [np.array(path) for path in sorted_paths]
    return sorted_paths

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




import socket
import select
import threading
import queue

class PlotterClient:
    ''' Plots to a remote instance of axidraw_server.py'''
    def __init__(self, address_or_settings='localhost', port=80, home_pos=[0,0], raw=False, use_feedrate=False):
        """
        :param address_or_settings:  (Default value = './client_settings.json') This parameter can specify either an IP address for a server,
        or the path to a json file containing the connection settings. The json file must contain the entries 'address' and 'port' with
        the IP address and the port number of the server.
        :param port:  (Default value = None) This parameter is required if adress_or_settings explicitly defines an IP address.
        If a json settings file is specified, defining this parameter will override the port defined in the file.
        :param raw:  (Default value = False): This will send coordinates (unscaled) to the server that will not be automatically scaled.
        Avoid using unless the coordinate system of the drawing fits in the drawing area.
        """
        if isinstance(address_or_settings, str) and '.json' in address_or_settings:
            import json
            try:
                settings = json.loads(open(address_or_settings).read())
                self.address = settings['address']
                self.port = settings['port']
            except FileNotFoundError as e:
                print(e)
                self.address = None
                self.port = port
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
        self.home_pos = home_pos

        # --- Listener state (single reader model) ---
        self._listener_thread = None
        self._listener_running = threading.Event()
        self._recv_buf = ""                # string buffer to assemble lines
        self._on_hash_line = None          # callable(line: str) | None
        self._lines = queue.SimpleQueue()  # optional: all complete lines

        self.open()
        # # Kick off a drawing session if requested
        # if use_feedrate:
        #     self.drawing_start_feed()
        # else:
        #     self.drawing_start()

    # ------------- Context manager -------------
    def __enter__(self):
        #self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False  # bubble exceptions

    # ------------- Connection management -------------
    def open(self):
        if self.address is None:
            return
        server_address = (self.address, self.port)
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect(server_address)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print("Openng socket")
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
        # Stop the listener first so it's not reading while we close
        self.stop_listening()
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.sock.close()
            finally:
                self.sock = None
        self.socket_open = False
        self.paths.clear()

    # ------------- Non-blocking listener API -------------
    def on_hash(self, callback):
        """
        Register a callback(line: str) that is invoked for each complete
        line received from the socket that starts with '#'.
        Pass None to disable.
        """
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        self._on_hash_line = callback

    def start_listening(self, callback=None, poll_interval=0.1):
        """
        Start a background listener thread that is the SOLE reader of the socket.
        It collects complete lines and calls `callback` for lines beginning with '#'.
        """
        if callback is not None:
            self.on_hash(callback)

        #self.open()  # ensure connected
        if not self.socket_open:
            return

        if self._listener_thread and self._listener_thread.is_alive():
            return  # already running

        self._listener_running.set()
        self._listener_thread = threading.Thread(
            target=self._listen_loop, args=(poll_interval,), daemon=True
        )
        self._listener_thread.start()

    def stop_listening(self):
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_running.clear()
            self._listener_thread.join(timeout=0.5)

    def _listen_loop(self, poll_interval):
        while self._listener_running.is_set():
            if not self.socket_open or self.sock is None:
                # sleep a tick and try again
                select.select([], [], [], poll_interval)
                continue

            # Only read when data is ready
            rlist, _, _ = select.select([self.sock], [], [], poll_interval)
            if not rlist:
                continue

            try:
                chunk = self.sock.recv(4096)
            except (BlockingIOError, InterruptedError):
                continue
            except OSError:
                # socket likely closed or errored
                self.socket_open = False
                continue

            if not chunk:
                # peer closed
                self.socket_open = False
                break

            # Decode and append to buffer
            try:
                text = chunk.decode("utf-8", errors="replace")
            except Exception:
                text = chunk.decode("latin-1", errors="replace")

            self._recv_buf += text

            # Emit complete lines
            while True:
                nl = self._recv_buf.find("\n")
                if nl == -1:
                    break
                line = self._recv_buf[:nl].rstrip("\r")
                self._recv_buf = self._recv_buf[nl + 1:]

                # store for optional consumers
                self._lines.put(line)

                # fire callback on hash-prefixed lines
                cb = self._on_hash_line
                if cb and line.startswith("#"):
                    try:
                        cb(line)
                    except Exception as e:
                        # never let user callback crash the loop
                        print(f"on_hash callback error: {e}")

    # ------------- Send helpers -------------
    def send(self, msg):
        """
        Sends a UTF-8 message. If the listener is running, we won't auto-close
        after an auto-open to avoid racing the reader.
        """
        listener_active = self._listener_thread and self._listener_thread.is_alive()
        if not self.socket_open:
            self.open()
            auto_open = not listener_active
        if self.socket_open:
            self.sock.sendall(msg.encode('utf-8'))

    def sendln(self, msg):
        self.send(msg + '\n')

    # ------------- Drawing lifecycle -------------
    def drawing_start(self, title=''):
        if title:
            self.sendln('PATHCMD title ' + title)
        self.sendln('PATHCMD drawing_start')

    def drawing_start_feed(self, title=''):
        if title:
            self.sendln('PATHCMD title ' + title)
        self.sendln('PATHCMD drawing_start_feed')

    def drawing_end(self, close=False):
        print("Drawing end")
        if self.raw:
            self.drawing_end_raw()
            return
        self.sendln('PATHCMD drawing_end')
        if close:
            raise ValueError("Should not close")
            self.close()

    def drawing_end_raw(self):
        self.sendln('PATHCMD drawing_end_raw')


    # ------------- Path ops -------------
    def draw_paths(self, S, title='', close=False, has_feedrate=False):
        try:
            if has_feedrate:
                self.drawing_start_feed(title)
            else:
                self.drawing_start(title)
            for P in S:
                if isinstance(P, str):
                    print('sending cmd ' + P)
                    self.sendln('PATHCMD cmd ' + P)
                else:
                    print('sending path ')
                    if has_feedrate and (np is not None):
                        try:
                            print('min feed', np.min(P[:, -1]))
                        except Exception:
                            pass
                    self.add_path(P, has_feedrate=has_feedrate)
            self.drawing_end(close)
        except ConnectionRefusedError as e:
            print('could not connect to network')
            print(e)

    def drawing(self, drawing, title=''):
        try:
            self.drawing_start(title)
            for P in getattr(drawing, "paths", []):
                self.add_path(P)
            self.drawing_end()
        except ConnectionRefusedError as e:
            print('could not connect to network')
            print(e)

    def add_path(self, P, has_feedrate=False):
        if not len(P):
            return
        # path_to_str is assumed to exist elsewhere in your codebase
        if has_feedrate:
            if len(P[0]) == 3:
                self.sendln('PATHCMD fstroke %d %s' % path_to_str(P))
            elif len(P[0]) == 4:
                self.sendln('PATHCMD fstroke3 %d %s' % path_to_str(P))
        else:
            if len(P[0]) == 2:
                self.sendln('PATHCMD stroke %d %s' % path_to_str(P))
            elif len(P[0]) == 3:
                self.sendln('PATHCMD stroke3 %d %s' % path_to_str(P))

    def draw_path(self, P, has_feedrate=False):
        if not len(P):
            return
        if has_feedrate:
            if len(P[0]) == 3:
                self.sendln('PATHCMD path %d %s' % path_to_str(P))
            elif len(P[0]) == 4:
                self.sendln('PATHCMD path %d %s' % path_to_str(P))
        else:
            if len(P[0]) == 2:
                self.sendln('PATHCMD path %d %s' % path_to_str(P))
            elif len(P[0]) == 3:
                self.sendln('PATHCMD path %d %s' % path_to_str(P))

    # ------------- Device controls -------------
    def motors_off(self):
        self.sendln('OFF')

    def motors_on(self):
        self.sendln('ON')

    def goto(self, pos):
        self.sendln(f'PATHCMD goto {pos[0]} {pos[1]}')

    def pen_up(self):
        self.sendln('PATHCMD pen_up')

    def pen_to(self, val):
        self.sendln(f'PATHCMD pen_to {val}')

    def pen_down(self):
        self.sendln('PATHCMD pen_down')

    def home(self):
        self.sendln('PATHCMD home')

    def feedrate(self, amt):
        self.sendln('PATHCMD feedrate %d' % (int(amt)))

    # ------------- Plot interface compatibility -------------
    def _set_bounds(self, w, h):
        self.bounds = (w, h)

    def _stroke(self, P):
        self.paths.append(P)

    # ------------- Removed blocking APIs -------------
    def wait(self):
        raise RuntimeError("wait() removed: the listener thread is the sole socket reader now.")








def path_to_str(P):
    ''' Convert a path to a (num_points, point sequence) tuple'''
    #if type(P) == np.ndarray:
    #    P = P.T
    dim = len(P[0])
    fmt = ' '.join(['%f' for _ in range(dim)])
    return len(P), ' '.join([fmt%tuple(p) for p in P])
    # if len(P[0]) == 2:
    #     return len(P), ' '.join(['%f %f'%(p[0], p[1]) for p in P])
    # return len(P), ' '.join(['%f %f %f'%(p[0], p[1], p[2]) for p in P])

AxiDrawClient = PlotterClient

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
