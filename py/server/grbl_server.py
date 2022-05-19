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
args.add_argument('--device', type=str, default='default', #/dev/tty.usbserial-10',
                 help='''Serial device''')
args.add_argument('--port', type=int, default=80,
                 help='''Server port number''')
args.add_argument('--nx', type=int, default=1,
                 help='''Number of horizontal subdivisions for test plots''')
args.add_argument('--ny', type=int, default=1,
                 help='''Number of vertical subdivisions for test plots''')
args.add_argument('--padding', type=float, default=0.2,
                 help='''Padding in inches for subdivided drawing''')
args.add_argument('--size', type=float, default=420,
                 help='''Reference work area size in mm''')
args.add_argument('--y_up', type=bool, default=False,
                 help='''If true this indicates that the input drawing has origin in the bottom left''')
args.add_argument('--start_index', type=int, default=0,
                 help='''Default start index''')
args.add_argument('--format', type=str, default='A3',
                 help='''Paper format (A4,A5,A3)''')
args.add_argument('--ox', type=float, default=299, help='Origin x (mm)')
args.add_argument('--oy', type=float, default=300, help='Origin y (mm)')
# args.add_argument('--pu', type=float, default=-32,
#                  help='''Pen up distance (lower value is lower)''')
# args.add_argument('--pd', type=float, default=-47,
#                  help='''Pen down distance (lower value is lower)''')

args.add_argument('--pd', type=float, default=-45,
                 help='''Pen up distance (lower value is lower)''')
args.add_argument('--pu', type=float, default=-35,
                 help='''Pen down distance (lower value is lower)''')


cfg = args.parse_args()
print(sys.argv)

RX_BUFFER_SIZE = 100 #128

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

class GrblDevice(threading.Thread):
    def __init__(self):
        self.devices = ['/dev/tty.usbserial-10',
                   '/dev/tty.usbserial-110']
        if cfg.device != 'default':
            self.devices = [cfg.device]

        threading.Thread.__init__(self)
        self.alive = True
        self.chunks = []
        self.s = None
        self.cmd_count = 0

    def run(self):
        no_osc = False
        for dev in self.devices:
            try:
                print('trying ' + dev)
                s = serial.Serial(dev, 115200)
                print('successfully connected to device :' + dev)
                break
            except (FileNotFoundError, serial.serialutil.SerialException) as e:
                print('could not connect to ' + dev)

        self.s = s
        #self.s = serial.Serial(cfg.device, 115200)
        #s = self.s
        verbose = True

        # Wake up grbl
        print("Initializing grbl...")
        s.write("\r\n\r\n".encode())
        # Wait for grbl to initialize and flush startup text in serial input
        time.sleep(2)
        s.flushInput()

        ack = False
        self.home()

        # periodic() # Start status report periodic timer
        while self.alive:
            # time.sleep(0.2)
            # continue

            while self.chunks:
                time.sleep(1.0/30)
                lock.acquire()
                lines = self.chunks[0]
                # print('Lines: ')
                # print(lines)

                l_count = 0
                g_count = 0
                c_line = []
                error_count = 0
                for line in lines:
                    # time.sleep(10.0/1000)
                    l_block = line.strip()
                    if not l_block.isspace() and len(l_block) > 0:
                        s.write((l_block + '\n').encode()) # Send block to grbl
                        self.cmd_count += 1
                        res = s.readline()
                        print(res)

                print('next chunk')
                print('cmd count %d'%self.cmd_count)
                self.chunks.pop(0)
                lock.release()

                s.flushInput()

                # while lines:
                #     #lock.acquire()
                #     line = lines[0] #.pop(0)
                #     print('parsing: ' + line)
                #     time.sleep(0.01)

                #     #lock.release()

                #     l_count += 1 # Iterate line counter
                # #     l_block = re.sub('\s|\(.*?\)','',line).upper() # Strip comments/spaces/new line and capitalize
                #     l_block = line.strip()
                #     c_line.append(len(l_block)+1) # Track number of characters in grbl serial read buffer
                #     grbl_out = b''
                #     while sum(c_line) >= RX_BUFFER_SIZE | s.inWaiting() :
                #         if s.inWaiting():
                #             try:
                #                 print('readline')
                #                 out_temp = s.readline().strip() # Wait for grbl response
                #             except serial.SerialException as e:
                #                 print(e)
                #                 continue
                #         else:
                #             continue

                #         print('done reading: ')
                #         print(out_temp)
                #         if out_temp.find(b'ok') < 0 and out_temp.find(b'error') < 0 :
                #             print(("  Debug: ",out_temp)) # Debug response
                #         else:
                #             grbl_out += out_temp;
                #             g_count += 1 # Iterate g-code counter
                #             grbl_out += str(g_count).encode(); # Add line finished indicator
                #             print('Tmp :')
                #             print(grbl_out)
                #             del c_line[0]
                #     print('exit loop')
                #     if verbose: print("SND: " + str(l_count) + " : " + l_block)
                #     s.write((l_block + '\n').encode()) # Send block to grbl
                #     lines.pop(0)
                #     if verbose : print(("BUF:",str(sum(c_line)),"REC:",grbl_out))
                    #print('next')

        print('Closing device')
        # Close file and serial port
        #f.close()
        s.close()

    def pen_up(self, newchunk=True):
        if newchunk:
            lock.acquire()
            self.chunks.append([])
        self.chunks[-1].append(f'G0 Z{cfg.pu}')
        if newchunk:
            lock.release()
    def pen_down(self, newchunk=True):
        if newchunk:
            lock.acquire()
            self.chunks.append([])
        self.chunks[-1].append(f'G0 Z{cfg.pd}')
        if newchunk:
            lock.release()

    def draw_path(self, P, newchunk=True):
        def pointstr(p):
            if len(p) == 2:
                return 'X%.2f Y%.2f'%(p[0], p[1])
            else:
                return 'X%.2f Y%.2f Z%.2f'%(p[0], p[1], p[2])


        if newchunk:
            lock.acquire()
            self.chunks.append([])
        #self.cmd_count = 0
        c = 0
        self.pen_up(False)
        c+=1
        for i, p in enumerate(P):
            if not i:
                #self.chunks[-1].append('G0 X%.2f Y%.2f'%(p[0], p[1]))
                self.chunks[-1].append('G0 ' + pointstr(p)) #X%.2f Y%.2f'%(p[0], p[1]))
                c+=1
                self.pen_down(False)
                c+=1
            else:
                #self.chunks[-1].append('G1 X%.2f Y%.2f F20000.0'%(p[0], p[1])) #F2700.0
                self.chunks[-1].append('G1 ' + pointstr(p) + ' F20000.0')
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

    def command(self, cmd):
        lock.acquire()
        self.chunks.append([])
        self.chunks[-1].append(cmd)
        lock.release()

device = GrblDevice()
device.start()


# V3_SIZEX and V3_SIZEY
# - this value is NOT the size of your paper
# - this value DOES control how PATHCMD scales x and y to your paper size
# - x,y coordinate values passed to PATHCMD get mapped into to this range
#   - with 0.0 mapped to 0 (closest to the USB connector in both axes)
#   - and  1.0 mapped to V3_SIZEX or V3_SIZEY
# - you can use coordinates >1.0 no problem
#   - they simply extrapolate beyond V3_SIZEX or V3_SIZEY
# - so with an 8.5 x 11 inch paper you might choose V3_SIZEX==V3_SIZEY==8.5
#   - with x coordinates > 1.0 to reach the extra 2.5 inches
# - it's important that V3_SIZEX==V3_SIZEY if NX==NY==1
#   - otherwise your PATHCMD drawing will be stretched or squished in one axis
# - if you set NX and NY to values other than 1
#   - V3_SIZEX and V3_SIZEY must have the opposite proportion to avoid squishing
# - for more details, see https://lurkertech.com/axiserver
#
V3_SIZEX = cfg.size #8.5
V3_SIZEY = cfg.size #8.5
#
if cfg.format != 'none':
    paper_sizes = {
        'A4': (297, 210), #21 x 29.7 cm
        'A3': (420, 297), #297 x 420 mm
        'A2': (594, 420),
        'A5': (210, 148.5) #148.5 x 210 mm
    }
    sz = paper_sizes[cfg.format][1]
    V3_SIZEX, V3_SIZEY = sz, sz

# - sets the division of the plotter surface
# - each new plot goes into the next place on the paper
#   - helpful when you're debugging some plotting code
#   - lets you avoid constantly putting new paper or manually moving the pen
# - to plot on the whole surface set these to 1
# - if these are 1, V3_SIZEX should equal V3_SIZEY
# - if these are not 1, V3_SIZEX/V3_SIZEY must stay in proportion
#   - for more details, see https://lurkertech.com/axiserver
#
NX = cfg.nx
NY = cfg.ny
PADDING = cfg.padding

# DB: Adjust size to avoid "squishing"
if NX != NY:
    V3_SIZEX = (NX/NY) * V3_SIZEX

# how the plot area is divided
CELLSIZEX = V3_SIZEX/NX
CELLSIZEY = V3_SIZEY/NY
POSX = 0
POSY = 0

print('Settings:')
print('V3 SIZE: %.3f'%(V3_SIZEX))
print('subdivision (%d, %d)'%(NX, NY))
print('CELL SIZE (%.3f, %.3f)'%(CELLSIZEX, CELLSIZEY))


def nextpos():
    global POSX
    global POSY
    POSX = POSX+1;
    if(POSX == NX):
        POSX = 0
        POSY = POSY+1
        if(POSY == NY):
            POSY = 0

for i in range(cfg.start_index):
    nextpos()

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

# DB: Note, addition here. We want to be able to draw with specific coordinates (in inches)
# which can be done by using the "drawing_end_raw" command
def drawing_end(raw=False):
    global paths
    #global title
    #d = axi.Drawing(paths)
    #text_pos = (PADDING, V3_SIZEX-PADDING)
    if not raw:
        mat = np.eye(3)
        if cfg.y_up:
            mat = geom.scaling_2d([1,-1])

        S = geom.affine_transform(mat, [np.array(P) for P in paths])
        src_rect = geom.bounding_box(S)
        dst_rect = geom.make_rect(cfg.ox + POSX*CELLSIZEX, cfg.oy + POSY*CELLSIZEY, CELLSIZEX, CELLSIZEY)
        mat = geom.rect_in_rect_transform(src_rect, dst_rect, PADDING)
        S = geom.affine_transform(mat, S)
        nextpos()
    else:
        S = [np.array(P) for P in paths]
        print(S)
    # if title:
    #     font = axi.Font(axi.FUTURAL, 7.5) #
    #     dtext = font.text(title)
    #     dtext
    #     dtext = dtext.translate(*text_pos)
    #     d.add(dtext)
    print('Sending to device:')
    #print(S)
    device.draw_paths(S)

    title = '' # Reset title

    print("DRAWING END")
    print("")

def pathcmd(*ary):
    #print(ary)
    if ary[1] == "drawing_start":
        print("PCMD: start")
        drawing_start()
    elif ary[1] == "drawing_end":
        print("PCMD: end")
        drawing_end(False)
    elif ary[1] == "drawing_end_raw": # DB: added raw drawing
        print("PCMD: end raw")
        drawing_end(True)
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
    elif ary[1] == 'home':
        device.home()
        print('received home')
    elif ary[1] == 'title':
        set_title(' '.join(ary[2:]))
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
                pathcmd(*ary)
            elif ary[0] == 'wait':
                device.wait()
                resp = 'done\n'.encode('utf-8')
                connection.send(resp)
            else:
                response = device.command(*ary)
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
print('end')
