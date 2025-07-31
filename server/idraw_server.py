#!/usr/bin/env python3

#
#       axiserver.py - tiny AxiDraw plot server
#
#               Chris Pirazzi and Paul Haeberli
#
# For details and instructions, see https://lurkertech.com/axiserver
#
# some simple additions by Daniel Berio

import threading
import serial
import re
import time
import sys
import argparse
import threading
import polygonsoup.geom as geom
import numpy as np

# DB: added arguments
import argparse
args = argparse.ArgumentParser(description='Grbl server')
args.add_argument('--device', type=str, default='default',
                 help='''Serial device''')
args.add_argument('--port', type=int, default=80,
                 help='''Server port number''')
args.add_argument('--baudrate', type=int, default=115200,
                 help='''Device baudrate''')
args.add_argument('--feedrate', type=int, default=3000)
args.add_argument('--feedrate_move', type=int, default=5000)
args.add_argument('--feedrate_pen', type=int, default=1000)

args.add_argument('--homing', type=int, default=0)
args.add_argument('--pd', type=float, default=7, #-45,
                 help='''Pen up distance (higher value is lower)''')
args.add_argument('--pu', type=float, default=0,
                 help='''Pen down distance (lower value is higher, 0 is the minimum)''')


cfg = args.parse_args()
print(sys.argv)

RX_BUFFER_SIZE = 128 # 100 #128
MAX_PD = 9.0

lock = threading.Lock()

state = lambda: None
state.pos = np.zeros(3)

def wait_for_idle(s, name=''):
    print(name + ': Waiting for plotter to be idle...')
    while True:
        s.write(b"?")  # Request GRBL status
        status = s.readline().decode('utf-8').strip()
        if '<Idle' in status:  # Ensure GRBL is idle before system commands
            break
    print('Plotter is idle, ready for commands.')


def wait_for_response(s, name='', timeout=60):
    print('waiting for response ' + name)
    ts = time.time()
    while time.time()-ts < timeout:
        if s.inWaiting():
            stuff = s.readline().strip()
            # print(str(stuff))
            if stuff.find(b'ok') >= 0:
                print("Reply is OK")
            break
    print('response time out')

import sys, select

class KeyListener(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.key = '' #None
        self.alive = True

    def pressed(self, c):
        if not self.key:
            return False

        if self.key[0] == 'p' or self.key[0] == ' ': #ord(c):
            self.key = ''
            return True
        return False

    def run(self):
        while self.alive:
            put = select.select([sys.stdin], [], [], 1)[0]
            if put:
                self.key = sys.stdin.readline().rstrip()
                print('Pressed: ' + self.key)
            # dr,dw,de = select.select([sys.stdin], [], [], 0)

            # c = sys.stdin.read()
            # return dr != []
            # self.pressed = stdscr.getch()
            time.sleep(0.1)
        print('endend key')


def find_cnc_device(baudrate=115200):
    import serial.tools.list_ports
    # List all available serial ports
    ports = serial.tools.list_ports.comports()
    print(ports)
    for port in ports:
        print(f"Checking port: {port.device}")
        if "usb" in port.device.lower():
            # Attempt to open the port to verify it's responsive
            try:
                with serial.Serial(port.device, baudrate=baudrate, timeout=1) as ser:
                    # Adjust baudrate to match your machine's settings
                    ser.write(b"?")  # Example: send a query or any command your device would respond to
                    response = ser.read(64)  # Read some bytes from the device
                    print(f"Response from {port.device}: {response}")
                    # Assuming a valid response confirms this port
                    return port.device
            except Exception as e:
                print(f"Failed to open port {port.device}: {e}")
    print("No CNC/plotter device found.")
    return None



key = KeyListener()
key.start()




def send_grbl_command(s, command):
    wait_for_idle(s, 'send grbl')  # Ensure GRBL is idle before sending commands
    s.write((command + '\n').encode('utf-8'))
    response = s.readline().decode('utf-8').strip()
    print(f"System command response: {response}")

    # time.sleep(0.1)
    # while serial_conn.in_waiting:
    #     line = serial_conn.readline().decode().strip()
    #     print(f"< {line}")

def set_settings(s):
    # GRBL settings to update
    settings = {
    '$11':  0.01,      # Junction deviation
    '$120': 300.0, #1500.0,   # X-axis acceleration
    '$121': 300.0, #1000.0,   # Y-axis acceleration
    '$32': '0',         # Laser mode?
    # Optional feed rate tweaks:
    # '$110': '10000.0',
    # '$111': '8000.0'
    }

    time.sleep(2)  # Let GRBL initialize
    s.flushInput()

    print("Sending new GRBL settings...")
    for key, value in settings.items():
        cmd = f"{key}={value}"
        print(f"> {cmd}")
        send_grbl_command(s, cmd)

    #print("Done. Verifying settings with $$:")
    #send_grbl_command(s, '$$')


class GrblDevice(threading.Thread):
    def __init__(self):
        if cfg.device != 'default':
            self.devices = [cfg.device]
        else:
            dev = find_cnc_device(cfg.baudrate)
            if dev is None:
                raise ValueError("Could not find plotter")
            self.devices = [dev]

        threading.Thread.__init__(self)
        self.alive = True
        self.chunks = []
        self.s = None
        self.cmd_count = 0
        self.paused = False

    def test_pause(self):
        if key.pressed(' '):
            self.paused = not device.paused
            print('Device pause: ' + str(int(self.paused)))

    def run(self):
        no_osc = False
        for dev in self.devices:
            try:
                print('trying ' + dev)
                s = serial.Serial(dev, cfg.baudrate)
                print('successfully connected to device :' + dev)
                break
            except (FileNotFoundError, serial.serialutil.SerialException) as e:
                print('could not connect to ' + dev)

        self.s = s

        verbose = True

        # Wake up grbl
        print("Initializing grbl...")
        s.write("\r\n\r\n".encode())
        # Wait for grbl to initialize and flush startup text in serial input
        time.sleep(2)
        s.flushInput()

        ack = False
        set_settings(s)

        print("Reading settings")
        s.write(b'$$\n')
        time.sleep(0.1)
        # Read response
        while True:
            if s.in_waiting:
                line = s.readline().decode('ascii').strip()
                print(line)
            else:
                break

        self.command('G21') # millimeters
        self.command('G17') # Xy
        #self.command('G90') # Absolute
        self.feedrate(cfg.feedrate)
        time.sleep(0.1)
        print("Setting feedrate to ", cfg.feedrate)

        if cfg.homing:
            self.home()

        print("Waiting for GRBL to be idle...")

        # periodic() # Start status report periodic timer
        while self.alive:
            # time.sleep(0.2)
            # continue
            self.test_pause()
            if self.paused:
                time.sleep(0.1)
                print('Device paused, waiting for resume...')
                continue



            while self.chunks:
                time.sleep(1.0/30)
                self.test_pause()
                if self.paused:
                    continue

                lock.acquire()
                print('Reading chunk')
                wait_for_idle(s, 'Chunk ready')  # Ensure GRBL is idle before sending commands
                print('Chunks: %d'%len(self.chunks))
                lines = self.chunks[0]
                # print('Lines: ')
                # print(lines)

                l_count = 0
                g_count = 0
                c_line = []
                error_count = 0

                for line in lines:
                    # while self.paused:
                    #     time.sleep(0.5)
                    l_count += 1
                    l_block = line.strip()

                    c_line.append(len(l_block) + 1)
                    grbl_out = ''

                    while sum(c_line) >= RX_BUFFER_SIZE - 1 or s.in_waiting:
                        out_temp = s.readline().strip().decode('utf-8')
                        if 'ok' not in out_temp and 'error' not in out_temp:
                            print("  Debug: ", out_temp)
                        else:
                            grbl_out += out_temp
                            g_count += 1
                            grbl_out += str(g_count)
                            if c_line:
                                del c_line[0]

                    
                    verbose = True
                    if verbose:
                        print(f"{l_count-1} >>> {l_block}", end=' ')
                    s.write((l_block + '\n').encode('utf-8'))
                    if verbose:
                        print(f"--- BUF: {sum(c_line)} REC: {grbl_out}")


                    # if not l_block.isspace() and len(l_block) > 0:
                    #     s.write((l_block + '\n').encode())  # Send to GRBL

                    #     # Wait for GRBL to respond with 'ok' or 'error'
                    #     while True:
                    #         res = s.readline().decode().strip()
                    #         if res == 'ok':
                    #             break
                    #         elif 'error' in res:
                    #             print(f"GRBL Error: {res}")
                    #             error_count += 1
                    #             break
                    #         elif res:  # Unexpected but printable
                    #             print(f"GRBL response: {res}")
                    #     #time.sleep(0.005)
                # for line in lines:
                #     while self.paused:
                #         time.sleep(0.5)

                #     # time.sleep(10.0/1000)
                #     l_block = line.strip()
                #     if not l_block.isspace() and len(l_block) > 0:
                #         s.write((l_block + '\n').encode()) # Send block to grbl
                #         self.cmd_count += 1
                #         res = s.readline()
                #         #print(res)

                #print('next chunk')
                #print('cmd count %d'%self.cmd_count)
                self.chunks.pop(0)
                lock.release()

                #s.flushInput()


        print('Closing device')
        # Close file and serial port
        #f.close()
        s.close()


    def pen_to(self, val):
        val = np.clip(float(val), 0.0, MAX_PD)
        lock.acquire()
        self.chunks.append(['G1G91 Z%.2f'%(val - state.pos[2])]) # F{cfg.feedrate_pen}')
        state.pos[2] = val
        lock.release()
        #wait_for_idle(self.s, False)  # Ensure GRBL is idle after pen up
        #
    def goto(self, pos):
        lock.acquire()
        pos = np.array(pos)
        pos[1] = -pos[1]  # Invert Y coordinate for GRBL compatibility
        d = pos - state.pos[:2]

        self.chunks.append(['G1G91 X%.2f Y%.2f'%(d[0], d[1])]) # F{cfg.feedrate_pen}')
        state.pos[:2] = pos[:2]  # Update X, Y position
        lock.release()
        #wait_for_idle(self.s, False)  # Ensure GRBL is idle after pen up
        #
    def pen_down(self, newchunk=True):
        if newchunk:
            lock.acquire()
            self.chunks.append([])
        self.chunks[-1].append(f'G0G91 Z{cfg.pd - state.pos[2]}') # F{cfg.feedrate_pen}')
        state.pos[2] = cfg.pd
        if newchunk:
            lock.release()
        #wait_for_idle(self.s, False)  # Ensure GRBL is idle after pen up

    def pen_up(self, newchunk=True):
        if newchunk:
            lock.acquire()
            self.chunks.append([])
        self.chunks[-1].append(f'G0G91 Z{cfg.pu - state.pos[2]}') # F{cfg.feedrate_pen}')
        state.pos[2] = cfg.pu
        if newchunk:
            lock.release()
        #wait_for_idle(self.s, False)  # Ensure GRBL is idle after pen up

    def feedrate(self, f):
        # if f < 1:
        #     print('Minimum feedrate is 50')
        #     f = 1
        # if f > 100:
        #     print('Maximum feedrate is 2500')
        #     f = 100
        self.command('F%d'%int(f))

    def draw_path_feedrate(self, P, newchunk=True):
        raise ValueError('draw_path_feedrate is not implemented, use draw_path instead')
        def pointstr(p):
            #print(len(p))
            #print(p)
            if len(p) == 3:
                return 'X%.2f Y%.2f F%d'%(p[0], -p[1], int(p[-1]))
            elif len(p) == 2:
                return 'X%.2f Y%.2f'%(p[0], -p[1])
            else:
                return 'X%.2f Y%.2f Z%.2f F%d'%(p[0], -p[1], max(0, min(p[2], MAX_PD)), int(p[-1]))

        if newchunk:
            lock.acquire()
            self.chunks.append([])
        #self.cmd_count = 0
        c = 0
        self.pen_up(False)
        c+=1
        for i, p in enumerate(P):
            if not i:
                # Move to first point 2d no feedrate
                self.chunks[-1].append('G0 ' + pointstr(p[:2])) # + ' F%d'%cfg.feedrate_move) #X%.2f Y%.2f'%(p[0], p[1]))

                #self.chunks[-1].append('G0 X%.2f Y%.2f'%(p[0], p[1]))
                # Skip feedrate on first move
                self.chunks[-1].append('G1 ' + pointstr(p)) #X%.2f Y%.2f'%(p[0], p[1]))
                c+=1
                if len(p) < 4:
                    self.pen_down(False)
                c+=1
            else:
                #self.chunks[-1].append('G1 X%.2f Y%.2f F20000.0'%(p[0], p[1])) #F2700.0
                self.chunks[-1].append('G1 ' + pointstr(p)) # + ' F20000.0')
                #print(self.chunks[-1])
                c+=1
        self.pen_up(False)
        c+=1
        if newchunk:
            lock.release()
        print('Desired cmd count: %d'%(c))
        return c

    def draw_path(self, P, newchunk=True):
        P = np.array(P)
        P[:,1] = -P[:,1]  # Invert Y coordinate for GRBL compatibility
        if P.shape[1] > 2:
            P[:,2] = np.clip(P[:,2], 0, MAX_PD)  # Clip Z values to max pen up distance

        def pointstr(p):
            if len(p) == 2:
                return 'X%.3f Y%.3f'%(p[0], p[1])
                #return f'X{p[0]} Y{p[1]}' #
            else:
                return 'X%.3f Y%.3f Z%.3f'%(p[0], p[1], p[2]) #max(0, min(p[2], MAX_PD)))
                #return f'X{p[0]} Y{p[1]} Z{max(0, min(p[2], MAX_PD))}' #
            #
        if newchunk:
            lock.acquire()
            self.chunks.append([])
            lock.release()

        #self.cmd_count = 0
        c = 0

        self.pen_up(newchunk)
        c+=1
        D = np.diff(np.vstack([state.pos[:P.shape[1]], P]), axis=0)  # Calculate differences between consecutive points

        for i, d in enumerate(D):
            if not i:
                if len(d) > 2:
                    self.chunks[-1].append('G1G91 ' + pointstr(d[:2])) # + ' F%d'%cfg.feedrate_move) #X%.2f Y%.2f'%(p[0], p[1]))
                    #state.pos[2] += d[2]  # Update Z position
                #self.chunks[-1].append('G0 X%.2f Y%.2f'%(p[0], p[1]))
                self.chunks[-1].append('G1G91 ' + pointstr(d)) #X%.2f Y%.2f'%(p[0], p[1]))
                state.pos[:len(d)] += d[:len(d)]  # Update X, Y a optionally Z position
                c+=1
                if len(d) < 3:
                    self.pen_down(newchunk)
                c+=1
            else:
                #self.chunks[-1].append('G1 X%.2f Y%.2f F20000.0'%(p[0], p[1])) #F2700.0
                self.chunks[-1].append('G1G91 ' + pointstr(d)) # + ' F20000.0')
                state.pos[:len(d)] += d[:len(d)]  # Update X, Y a optionally Z position
                c+=1

        self.pen_up(newchunk)
        c+=1
        #if newchunk:
        #    lock.release()
        print('Desired cmd count: %d'%(c))
        return c

    def draw_paths(self, paths, feedrate=False):
        lock.acquire()
        self.chunks.append([])
        self.cmd_count = 0
        draw_path = self.draw_path
        if feedrate:
            draw_path = self.draw_path_feedrate
        c = 0
        for P in paths:
            if type(P) == str:
                self.command(P, False)
            else:
                c += draw_path(P, False)
        lock.release()
        print('Desired cmd count: %d'%c)

        chunks = sum(self.chunks, [])
        f = open('test.gcode', 'w')
        f.write('\n'.join(chunks))
        f.close()
        #print('\n'.join(chunks))


    def motors_off(self):
        lock.acquire()
        self.s.write(b'$1=0\n')
        wait_for_response(self.s, 'motors off', 10.0)
        time.sleep(0.1)
        # relative move of zero
        self.s.write(b'G91\n')
        self.s.write(b'G0 X0 Y0 Z0\n')
        # absolute
        self.s.write(b'G91\n')
        print('motors off')
        time.sleep(0.2)
        wait_for_response(self.s, 'motors off', 10.0)
        lock.release()

    def null_move(self, delay=0.1):
        # Dummy move to re-energize motors
        self.s.write(b'G91\n')
        self.s.write(b'G0 X0.01\n')
        self.s.write(b'G0 X-0.01\n')
        self.s.write(b'G91\n')
        #self.s.flush()
        time.sleep(delay)
        wait_for_response(self.s, 'null move', 10.0)

    def motors_switch(self, state):
        ''' Turn motors on/off
        '''
        # Hacky GRBL version, since we don't have a way to turn off motors explictly
        lock.acquire()

        # Set idle delay to "never turn off"
        if state:
            self.s.write(b'$1=255\n')
        else:
            self.s.write(b'$1=0\n')
        time.sleep(0.2)
        self.s.flush()
        wait_for_response(self.s, 'set hold delay', 10.0)

        self.null_move()

        # Reset current position
        self.s.write(b'G92 X0 Y0 Z0\n')
        time.sleep(0.2)
        self.s.flush()

        if state:
            print('motors on')
        else:
            print('motors off')

        wait_for_response(self.s, 'set zero', 10.0)

        # # Optional: clear serial buffer
        # while self.s.in_waiting:
        #     print(self.s.readline().decode().strip())

        lock.release()

    def home(self, enable_motors=True):

        lock.acquire()
        self.chunks = [[]]

        print('Homing...')
        if enable_motors:
            self.s.write(b'$1=255\n')
            wait_for_idle(self.s, 'Home')
        self.s.write(b'$H\n')
        wait_for_response(self.s, 'homing', 60.0)
        self.s.write(b'G92 X0 Y0 Z0\n') #
        wait_for_response(self.s, 'set initial pos', 10.0)
        wait_for_idle(self.s, 'after homing')  # Ensure GRBL is idle after homing
        print('Done homing')
        lock.release()
        #self.chunks[-1].append('$H')
        #self.chunks[-1].append('G92 X0 Y0 Z0')

    def command(self, cmd, do_lock=True):
        if do_lock:
            lock.acquire()
            self.chunks.append([])
        self.chunks[-1].append(cmd)
        if do_lock:
            lock.release()

device = GrblDevice()
device.start()


def goodbuf(buf):
   if not buf: return False
   if buf[0] == 0xff: return False
   if buf == '': return False
   return True

sofar = ""

def recv(connection):
    global sofar
    while True:
        off = sofar.find("\n");
        if -1 != off: break
        #print("reading more from connection")
        buf = connection.recv(1024)
        if not buf: return None
        if buf[0] == 0xff: return None
        if buf == '': return None
        buf = buf.decode("utf-8")
        #print("read [" + buf + "] from connection")
        sofar += buf
    ret = sofar[0:off];
    ret = ret.rstrip()
    sofar = sofar[off+1:]
    #print("remaining sofar is [" + sofar + "]")
    #print("returning [" + ret + "] to caller")
    return ret;

curpath = []
paths = []
has_feedrate = False
title = ''

def set_title(txt):
    global title
    print('Setting title: ' + txt)
    title = txt

def stroke_start():
    global curpath
    curpath = []


def stroke_addpoint(p):
    global curpath
    curpath.append(p)

def stroke_end():
    global paths
    global curpath
    paths.append(np.array(curpath))

def drawing_start(feedrate=False):
    print("DRAWING START")
    global paths, has_feedrate
    paths = []
    has_feedrate = feedrate

def drawing_add_cmd(cmd):
    global paths
    paths.append(cmd)

# DB: Note, addition here. We want to be able to draw with specific coordinates (in inches)
# which can be done by using the "drawing_end_raw" command
def drawing_end():
    import matplotlib.pyplot as plt
    from polygonsoup import plut

    S = [np.array(P) if type(P) != str else P for P in paths]
    # plt.figure(figsize=(5, 5))
    # plut.stroke(S, 'k', lw=0.5, zorder=0)
    # plt.show()
    print('Sending to device:')
    device.draw_paths(S, has_feedrate)
    print("DRAWING END")
    print("")

def pathcmd(*ary):
    #print(ary)
    if "drawing_start" in ary[1]:
        print("PCMD: start")
        drawing_start()
    if "drawing_start_feed" in ary[1]:
        print("PCMD: start")
        drawing_start(True)
    elif "drawing_end" in ary[1]:
        print("PCMD: end")
        drawing_end()
    elif ary[1] == "stroke":
        print("PCMD: stroke")
        npoints = ary[2]
        #print("npoints" + npoints)
        stroke_start()
        for i in range(int(npoints)):
            x = float(ary[2*i+3])
            y = float(ary[2*i+4])
            #print("pointx: " + str(x) + "pointy: " + str(y))
            stroke_addpoint((x, y))
        stroke_end()

    elif ary[1] == "stroke3":
        print("PCMD: stroke3")
        npoints = ary[2]
        #print("npoints" + npoints)
        stroke_start()
        for i in range(int(npoints)):
            x = float(ary[3*i+3])
            y = float(ary[3*i+4])
            z = float(ary[3*i+5])
            #print("pointx: " + str(x) + "pointy: " + str(y))
            stroke_addpoint((x, y, z))
        stroke_end()
    elif ary[1] == "fstroke":
        print("PCMD: fstroke")
        npoints = ary[2]
        #print("npoints" + npoints)
        stroke_start()
        for i in range(int(npoints)):
            x = float(ary[3*i+3])
            y = float(ary[3*i+4])
            f = float(ary[3*i+5])
            #print("pointx: " + str(x) + "pointy: " + str(y))
            stroke_addpoint((x, y, f))
    elif ary[1] == 'path':
        print("PCMD: path")
        npoints = ary[2]
        #print("npoints" + npoints)
        stroke_start()
        for i in range(int(npoints)):
            x = float(ary[2*i+3])
            y = float(ary[2*i+4])

            #print("pointx: " + str(x) + "pointy: " + str(y))
            stroke_addpoint((x, y))
        stroke_end()
        device.draw_path(curpath)
    elif ary[1] == "fstroke3":
        print("PCMD: stroke3")
        npoints = ary[2]
        #print("npoints" + npoints)
        stroke_start()
        for i in range(int(npoints)):
            x = float(ary[4*i+3])
            y = float(ary[4*i+4])
            z = float(ary[4*i+5])
            f = float(ary[4*i+6])
            #print("pointx: " + str(x) + "pointy: " + str(y))
            stroke_addpoint((x, y, z, f))
        stroke_end()
    elif ary[1] == 'pen_up':
        device.pen_up()
        print('received pen up')
    elif ary[1] == 'pen_to':
        device.pen_to(ary[2])
        print('received pen to')
    elif ary[1] == 'goto':
        device.goto([float(ary[2]), float(ary[3])])
        print('received pen to')
    elif ary[1] == 'pen_down':
        device.pen_down()
        print('received pen down')
    elif ary[1] == 'feedrate':
        device.feedrate(int(ary[2]))
        print('received feedrate ', ary[2])
    elif ary[1] == 'feedrate':
        device.feedrate(float(ary[2]))
    elif ary[1] == 'home':
        device.home()
        print('received home')
    elif ary[1] == 'cmd':
        drawing_add_cmd(' '.join(ary[2:]))
    else:
        print("strange PATHCMD: " + ary[1])
        # if ary[1][0] == '$':
        #     print('sending command ' + ary[1][1:])
        #     device.command(ary[1][1:])

import socket,os
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Grbl server, binding socket to port %d'%(cfg.port))
sock.bind(('', cfg.port))  # CHANGE PORT NUMBER HERE!
sock.listen(5)

#import curses
#stdscr = curses.initscr()

try:
    while True:
        connection,address = sock.accept()
        sendit = lambda string: connection.send(string.encode("utf-8"))
        #print("got connection")
        sofar = "";
        while True:

            buf = recv(connection)
            if buf is None: break
            source = ""
            if(buf == "\""):
                #print("starting some python code")
                while True:
                    buf = recv(connection)
                    if buf is None: break
                    if(buf == "\""):
                        #print("doing exec of:\n" + source)
                        # DANGEROUS on public networks!!!!
                        # uncomment for python interface
                        #exec(source)
                        break
                    source += buf + "\n"
            if source != "": continue
            if buf is None: break
            ary = buf.split(" ")
            #print("cmd ary:")
            # print(ary)
            # print(ary[0])
            if ary[0] == "PATHCMD":
                #print(ary)
                pathcmd(*ary)
            elif ary[0] == 'OFF':
                device.motors_switch(0)
            elif ary[0] == 'ON':
                device.motors_switch(1)
            elif ary[0] == 'wait':
                device.wait()
                resp = 'done\n'.encode('utf-8')
                connection.send(resp)
            else:
                cmd = ' '.join(ary)
                if cmd == 'sleep':
                    #print('sleeping half sec')
                    time.sleep(0.5)
                else:
                    print('executing command: ' + cmd)
                    response = device.command(cmd)
                #print("device response: " + response)
                # response += "\n"
                # response = response.encode('utf-8')
                # connection.send(response)
        #print("connection closed")
        connection.close()

except KeyboardInterrupt:
    print('Stopping')

device.alive = False
time.sleep(1)
device.join()

key.alive = False
time.sleep(0.5)
key.join()

print('end')
