#!/usr/bin/env python3

#
#       axiserver.py - tiny AxiDraw plot server
#
#               Chris Pirazzi and Paul Haeberli
#
# For details and instructions, see https://lurkertech.com/axiserver
#
# some simple additions by Daniel Berio

import axi
import sys

# DB: added arguments
import argparse
args = argparse.ArgumentParser(description='Axidraw server')

args.add_argument('--port', type=int, default=4000,
                 help='''Server port number''')
args.add_argument('--nx', type=int, default=3,
                 help='''Number of horizontal subdivisions for test plots''')
args.add_argument('--ny', type=int, default=2,
                 help='''Number of vertical subdivisions for test plots''')
args.add_argument('--padding', type=float, default=0.2,
                 help='''Padding in inches for subdivided drawing''')
args.add_argument('--size', type=float, default=5.8,
                 help='''Reference work area size in inches''')
args.add_argument('--y_up', type=bool, default=False,
                 help='''If true this indicates that the input drawing has origin in the bottom left''')
args.add_argument('--start_index', type=int, default=0,
                 help='''Default start index''')
args.add_argument('--format', type=str, default='none',
                 help='''Paper format (A4,A5,A3)''')
args.add_argument('--plt', type=bool, default=False,
                 help='''If True use matplotlib debug view instead''')

cfg = args.parse_args()
print(sys.argv)

try:
    device = axi.Device()
except Exception as e:
    print(e)

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
        'A4': (11.7, 8.3),
        'A3': (16.5, 11.7),
        'A5': (8.3, 5.8)
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

def stroke_end():
    global paths
    global curpath
    paths.append(curpath)

def drawing_start():
    print("DRAWING START")
    global paths
    paths = []

# DB: Note, addition here. We want to be able to draw with specific coordinates (in inches)
# which can be done by using the "drawing_end_raw" command
def drawing_end(raw=False):
    global paths
    global title
    d = axi.Drawing(paths)
    text_pos = (PADDING, V3_SIZEX-PADDING)
    if not raw:
        if cfg.y_up:
            d = d.scale(1.0, -1.0)
        d = d.scale_to_fit(CELLSIZEX, CELLSIZEY, PADDING)
        d = d.translate(POSX*CELLSIZEX,  POSY*CELLSIZEY)
        text_pos = (POSX*CELLSIZEX, (POSY+1)*CELLSIZEY-PADDING)
        nextpos()

    if title:
        font = axi.Font(axi.FUTURAL, 7.5) #
        dtext = font.text(title)
        dtext
        dtext = dtext.translate(*text_pos)
        d.add(dtext)

    axi.draw(d)

    title = '' # Reset title

    print("DRAWING END")
    print("")

def pathcmd(*ary):
    print(ary)
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
        print("npoints" + npoints)
        stroke_start()
        for i in range(int(npoints)):
            x = float(ary[2*i+3])
            y = float(ary[2*i+4])
            print("pointx: " + str(x) + "pointy: " + str(y))
            stroke_addpoint(x, y)
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


import socket,os
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('AxiDraw server, binding socket to port %d'%(cfg.port))
sock.bind(('', cfg.port))  # CHANGE PORT NUMBER HERE!
sock.listen(5)

print('entering loop')

while True:
    connection,address = sock.accept()
    sendit = lambda string: connection.send(string.encode("utf-8"))
    print("got connection")
    sofar = "";
    while True:
        buf = recv(connection)
        if buf is None: break
        print('received')
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
        print(ary)
        print(ary[0])

        if False:
            if ary[0] == "PATHCMD":
                pathcmd(*ary)
            elif ary[0] == 'wait':
                device.wait()
                resp = 'done\n'.encode('utf-8')
                connection.send(resp)
            else:
                response = device.command(*ary)
                print("device response: " + response)
                response += "\n"
                response = response.encode('utf-8')
                connection.send(response)
    #print("connection closed")
    connection.close()
