'''
  _   _   _   _   _   _   _   _   _   _   _
 / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
( P | O | L | Y | G | O | N | S | O | U | P )
 \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

Plotter-friendly graphics utilities
© Daniel Berio (@colormotor) 2021 - ...

utils - general Python utilitis (performance, files, strings)
'''


import pickle
import os, sys, time, shutil, requests
import argparse

import json
import numpy as np


class perf_timer:
    def __init__(self, name='', verbose=True):
        #if name and verbose:
        #    print(name)
        self.name = name
        self.verbose = verbose
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t)*1000
        if self.name and self.verbose:
            print('%s: elapsed time %.3f milliseconds'%(self.name, self.elapsed))


### Serialization
def load_pkl(data_path):
    with open(os.path.expanduser(data_path), "rb") as f:
        try:
            raw_data = pickle.load(f)
        except:
            try:
                print('failed normal load, trying latin1')
                raw_data = pickle.load(f, encoding='latin1')
            except:
                print('also failed latin1, trying bytes')
                raw_data = pickle.load(f, encoding='bytes')
    return raw_data


def save_pkl(X, data_path, protocol=4):
    with open(os.path.expanduser(data_path), "wb") as f:
        pickle.dump(X, f, protocol=protocol)


def load_json(path):
    import codecs
    try:
        with codecs.open(path, encoding='utf8') as fp:
            data = json.load(fp)
        return data
    except IOError as err:
        print(err)
        print ("Unable to load json file:" + path)
        return {}

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(data, path, encoder=NumpyEncoder):
    with open(path, 'w') as fp:
        #default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        #The above breaks
        json.dump(data, fp, indent=4, sort_keys=True, cls=encoder) #, default=default)

def json_string(data, encoder=NumpyEncoder):
    return json.dumps(data, cls=encoder)


def create_dir(dir):
    """create a directory if it doesn't exist. silently fail if it already does exist"""
    try:
        os.makedirs(dir)
    except OSError:
        if not os.path.isdir(dir):
            raise OSError
    return dir

def ensure_path(path):
    dir = os.path.dirname(path)
    create_dir(dir)
    return path

def ensure_abspath(dir):
    return os.path.abspath(create_dir(dir))

def delete_dir_contents(path):
    shutil.rmtree(path)
    os.mkdir(path)

def files_in_dir(path, exts=[], nohidden=True):
    files = []
    path = os.path.expanduser(path)
    for file in os.listdir(path):
        if (nohidden and
            os.path.basename(file).startswith('.')):
            continue
        if exts:
            if type(exts) != list:
                exts = [exts]
            for ext in exts:
                if file.endswith(ext):
                    files.append(os.path.join(path, file))
                    break
        else:
            files.append(os.path.join(path, file))

    return files


def filename_without_ext(path):
    return os.path.splitext(os.path.basename(path))[0]

def filename_ext(path):
    return os.path.splitext(os.path.basename(path))[1]

def filename(path):
    return os.path.basename(path)

def dirname(path):
    return os.path.dirname(path)

class FileWatcher(object):
    def __init__(self, paths):
        if type(paths) != list:
            paths = [paths]
        self._cached_stamps = [0 for _ in paths]
        self.filenames = paths

    def modified(self):
        res = False
        for i, filename in enumerate(self.filenames):
            stamp = os.stat(filename).st_mtime
            if stamp != self._cached_stamps[i]:
                self._cached_stamps[i] = stamp
                res = True
        return res



def process_exists(proc_name):
    ''' Returns True if process with proc_name exists
    from https://stackoverflow.com/questions/38056/how-to-check-if-a-process-is-still-running-using-python-on-linux'''
    import subprocess, re

    ps = subprocess.Popen("ps ax -o pid= -o args= ", shell=True, stdout=subprocess.PIPE)
    ps_pid = ps.pid
    output = ps.stdout.read().decode('utf8')
    ps.stdout.close()
    ps.wait()


    for line in output.split("\n"):
        res = re.findall("(\d+) (.*)", line)
        if res:
            pid = int(res[0][0])
            if (proc_name in res[0][1] and
                proc_name + '/' not in res[0][1] and # HACK In case a directory exists with same name
                pid != os.getpid() and pid != ps_pid):
                return True
    return False

def print_same_line(s):
    #sys.stdout.write("\r" + str(s))
    print(s, end='\r')
    #sys.stdout.flush()

def progress_bar(ratio, bar_len = 20):
    ''' Adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console'''
    sys.stdout.write("\r")
    progress = ""
    for i in range(bar_len):
        if i < int(bar_len * ratio):
            progress += chr(0x2588) #unichr(219) #'\xdb' #"="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, min(ratio * 100, 100)))
    sys.stdout.flush()

#!/usr/bin/env python3
import os, requests

cfg = lambda: None
cfg.dropbox_dir = '/transfer_box'
cfg.tmp_file = ''

def to_dropox(path, dropbox_dir='/transfer_box'):
    cfg.dropbox_dir = dropbox_dir
    cfg.tmp_file = os.path.abspath(path)
    return cfg.tmp_file

def dropbox_upload(token_path='~/dbox.txt'):
    TOKEN = open(os.path.expanduser(token_path)).read().strip()
    name = os.path.basename(cfg.tmp_file)
    create_dropbox_dir(cfg.dropbox_dir)
    dropbox_path = os.path.join(cfg.dropbox_dir, name)

    try:
        headers = {
            'Authorization': f'Bearer {TOKEN}',
            'Content-Type': 'application/octet-stream',
            'Dropbox-API-Arg': f'{{"path": "{dropbox_path}", "mode": "add"}}'
        }

        # Read the file content
        with open(cfg.tmp_file, 'rb') as file:
            file_content = file.read()

        # Make a request to the Dropbox API to upload the file
        response = requests.post(
            'https://content.dropboxapi.com/2/files/upload',
            headers=headers,
            data=file_content
        )

        # Check if the request was successful
        if response.status_code == 200:
            print("File uploaded to Dropbox successfully.")
        else:
            print(f"Failed to upload file to Dropbox: {response.status_code} - {response.text}")

        os.remove(cfg.tmp_file)

    except Exception as e:
        print("Error:", e)


def create_dropbox_dir(path, token_path='~/dbox.txt'):
    import os
    import dropbox
    from dropbox.exceptions import ApiError, AuthError
    TOKEN = open(os.path.expanduser(token_path)).read().strip()
    try:
        # Initialize Dropbox client
        dbx = dropbox.Dropbox(TOKEN)

        # Check if the directory already exists
        exists = False
        try:
            dbx.files_get_metadata(path)
            exists = True
        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                exists = False
            else:
                raise

        # Create the directory if it does not exist
        if not exists:
            dbx.files_create_folder_v2(path)
            print(f"Directory '{path}' created successfully.")
        else:
            print(f"Directory '{path}' already exists.")

    except AuthError as e:
        print("Authentication error:", e)
    except Exception as e:
        print("Error:", e)


''' Dropbox utilities usage
# Requires pip install dropbox
# And a file dbox.txt with the token in the home directory

if __name__ == "__main__":
    directory_path = '/transfer_box/igor'
    create_dropbox_dir(directory_path)

    f = open(to_dropox('./test.txt', '/transfer_box/igor'), 'w')
    f.write('Hello, Dropbox!')
    f.close()
    upload()

 '''
