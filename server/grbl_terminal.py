#!/usr/bin/env python3

import serial
import re
import time
import sys
import argparse
import threading
# import threading

import argparse
args = argparse.ArgumentParser(description='Axidraw server')

args.add_argument('--device', type=str, default='default',
                 help='''Serial device''')
args.add_argument('--port', type=int, default=80,
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
args.add_argument('--pu', type=float, default=-32,
                 help='''Pen up distance (lower value is lower)''')
args.add_argument('--pd', type=float, default=-47,
                 help='''Pen down distance (lower value is lower)''')

args.add_argument('--init', type=int, default=1, help="Indicates wether to init grbl or not")
cfg = args.parse_args()
print(sys.argv)


RX_BUFFER_SIZE = 128

lock = threading.Lock()

def wait_for_response(s):
    ts = time.time()
    while time.time()-ts < 10.0:
        read = False
        while s.inWaiting():
            stuff = s.readline().strip()
            print(str(stuff))
            read = True
            #if stuff.find(b'ok') >= 0:
            #    print("Reply is OK")
        if read:
            break

class GrblThread(threading.Thread):
    def __init__(self):
        self.devices = ['/dev/tty.usbserial-10',
                   '/dev/tty.usbserial-110']
        if cfg.device != 'default':
            self.devices = [cfg.device]

        threading.Thread.__init__(self)
        self.alive = True
        self.lines = []

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


        verbose = True

        # Wake up grbl
        if cfg.init:
            print("Initializing grbl...")
            s.write("\r\n\r\n".encode())
        wait_for_response(s)
        # Wait for grbl to initialize and flush startup text in serial input
        time.sleep(2)
        s.flushInput()

        l_count = 0
        g_count = 0
        c_line = []
        # periodic() # Start status report periodic timer
        while self.alive:
            if self.lines:
                #time.sleep(0.01)
                #lock.acquire()
                line = self.lines.pop(0)
                #lock.release()

                l_block = line.strip()
                if not l_block.isspace() and len(l_block) > 0:
                    s.write((l_block + '\n').encode()) # Send block to grbl
                    wait_for_response(s)
                    #res = s.readline()
                    #print(res)
        # Close file and serial port
        # f.close()
        s.close()

    def add_line(self):
        pass

grbl = GrblThread()
grbl.start()

try:
    print('Input commands')
    while True:
        print('> ', end="")
        cmd = input()
        if 'home' in cmd:
            grbl.lines.append('$H')
            grbl.lines.append('G92 X0 Y0 Z0')
        elif 'pu' in cmd:
            grbl.lines.append(f'G0 Z{cfg.pu}')
        elif 'pd' in cmd:
            grbl.lines.append(f'G0 Z{cfg.pd}')
        elif 'ud' in cmd:
            for i in range(3):
                grbl.lines.append(f'G0 Z{cfg.pd}')
                grbl.lines.append(f'G0 Z{cfg.pu}')

        else:
            grbl.lines.append(cmd)

except KeyboardInterrupt:
    print('Stopping')

grbl.alive = False
time.sleep(1)
grbl.join()
print('end')
