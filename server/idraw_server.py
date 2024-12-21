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
args.add_argument('--feedrate', type=int, default=100)
args.add_argument('--homing', type=int, default=0)
args.add_argument('--pd', type=float, default=6, #-45,
                 help='''Pen up distance (higher value is lower)''')
args.add_argument('--pu', type=float, default=0,
                 help='''Pen down distance (lower value is higher, 0 is the minimum)''')


cfg = args.parse_args()
print(sys.argv)

RX_BUFFER_SIZE = 100 #128
MAX_PD = 9.0

lock = threading.Lock()

def wait_for_response(s, name='', timeout=30):
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
        print("Setting millimeters")
        self.command('G21') # millimeters
        self.command('G17') # Xy
        self.command('G90') # Absolute
        self.feedrate(cfg.feedrate)
        time.sleep(0.1)
        print("Setting feedrate to ", cfg.feedrate)

        if cfg.homing:
            self.home()

        # periodic() # Start status report periodic timer
        while self.alive:
            # time.sleep(0.2)
            # continue
            self.test_pause()
            if self.paused:
                time.sleep(0.1)
                continue

            while self.chunks:
                time.sleep(1.0/30)
                self.test_pause()
                if self.paused:
                    continue

                lock.acquire()
                lines = self.chunks[0]
                # print('Lines: ')
                # print(lines)

                l_count = 0
                g_count = 0
                c_line = []
                error_count = 0
                for line in lines:
                    while self.paused:
                        time.sleep(0.5)

                    # time.sleep(10.0/1000)
                    l_block = line.strip()
                    if not l_block.isspace() and len(l_block) > 0:
                        s.write((l_block + '\n').encode()) # Send block to grbl
                        self.cmd_count += 1
                        res = s.readline()
                        #print(res)

                print('next chunk')
                print('cmd count %d'%self.cmd_count)
                self.chunks.pop(0)
                lock.release()

                s.flushInput()


        print('Closing device')
        # Close file and serial port
        #f.close()
        s.close()

    def pen_up(self, newchunk=True):
        if newchunk:
            lock.acquire()
            self.chunks.append([])
        self.chunks[-1].append(f'G1 Z{cfg.pu}')
        if newchunk:
            lock.release()

    def pen_down(self, newchunk=True):
        if newchunk:
            lock.acquire()
            self.chunks.append([])
        self.chunks[-1].append(f'G1 Z{cfg.pd}')
        if newchunk:
            lock.release()

    def feedrate(self, f):
        # if f < 1:
        #     print('Minimum feedrate is 50')
        #     f = 1
        # if f > 100:
        #     print('Maximum feedrate is 2500')
        #     f = 100
        self.command('F%d'%int(f))

    def draw_path(self, P, newchunk=True):
        def pointstr(p):
            if len(p) == 2:
                return 'X%.2f Y%.2f'%(p[0], -p[1])
            else:
                return 'X%.2f Y%.2f Z%.2f'%(p[0], -p[1], max(0, min(p[2], MAX_PD)))

        if newchunk:
            lock.acquire()
            self.chunks.append([])
        #self.cmd_count = 0
        c = 0
        self.pen_up(False)
        c+=1
        for i, p in enumerate(P):
            if not i:
                if len(p) > 2:
                    self.chunks[-1].append('G1 ' + pointstr(p[:2])) #X%.2f Y%.2f'%(p[0], p[1]))

                #self.chunks[-1].append('G0 X%.2f Y%.2f'%(p[0], p[1]))
                self.chunks[-1].append('G1 ' + pointstr(p)) #X%.2f Y%.2f'%(p[0], p[1]))
                c+=1
                if len(p) < 3:
                    self.pen_down(False)
                c+=1
            else:
                #self.chunks[-1].append('G1 X%.2f Y%.2f F20000.0'%(p[0], p[1])) #F2700.0
                self.chunks[-1].append('G1 ' + pointstr(p)) # + ' F20000.0')
                c+=1
        self.pen_up(False)
        c+=1
        if newchunk:
            lock.release()
        print('Desired cmd count: %d'%(c))
        return c

    def draw_paths(self, paths):
        lock.acquire()
        self.chunks.append([])
        self.cmd_count = 0
        c = 0
        for P in paths:
            if type(P) == str:
                self.command(P, False)
            else:
                c += self.draw_path(P, False)
        lock.release()
        print('Desired cmd count: %d'%c)

        chunks = sum(self.chunks, [])
        f = open('test.gcode', 'w')
        f.write('\n'.join(chunks))
        f.close()
        #print('\n'.join(chunks))

    def home(self):
        lock.acquire()
        self.chunks = [[]]
        self.s.write(b'$H\n')
        wait_for_response(self.s, 'homing', 60.0)
        self.s.write(b'G92 X0 Y0 Z0\n')
        wait_for_response(self.s, 'set initial pos', 10.0)
        #self.chunks[-1].append('$H')
        #self.chunks[-1].append('G92 X0 Y0 Z0')
        lock.release()

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
title = ''

def set_title(txt):
    global title
    print('Setting title: ' + txt)
    title = txt

def stroke_start():
    global curpath
    curpath = []

def stroke_addpoint(x, y):
    global curpath
    curpath.append((x, y))

def stroke_addpoint3(x, y, z):
    global curpath
    curpath.append((x, y, z))

def stroke_end():
    global paths
    global curpath
    paths.append(np.array(curpath))


def drawing_start():
    print("DRAWING START")
    global paths
    paths = []

def drawing_add_cmd(cmd):
    global paths
    paths.append(cmd)

# DB: Note, addition here. We want to be able to draw with specific coordinates (in inches)
# which can be done by using the "drawing_end_raw" command
def drawing_end():
    S = [np.array(P) if type(P) != str else P for P in paths]
    print('Sending to device:')
    device.draw_paths(S)
    print("DRAWING END")
    print("")

def pathcmd(*ary):
    #print(ary)
    if "drawing_start" in ary[1]:
        print("PCMD: start")
        drawing_start()
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
            stroke_addpoint(x, y)
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
            stroke_addpoint3(x, y, z)
        stroke_end()
    elif ary[1] == 'pen_up':
        device.pen_up()
        print('received pen up')
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
