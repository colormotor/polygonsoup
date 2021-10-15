
import pickle
import os, sys, time, shutil
import argparse

import json
import numpy as np


class perf_timer:
    def __init__(self, name=''):
        if name:
            print(name)
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.elapsed = (time.perf_counter() - self.t)*1000
        if self.name:
            print('%s: elapsed time %.3f'%(self.name, self.elapsed))


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


def save_pkl(X, data_path):
    with open(os.path.expanduser(data_path), "wb") as f:
        pickle.dump(X, f, protocol=2)


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
    def __init__(self, path):
        self._cached_stamp = 0
        self.filename = path

    def modified(self):
        stamp = os.stat(self.filename).st_mtime
        if stamp != self._cached_stamp:
            self._cached_stamp = stamp
            return True
        return False


def print_same_line(s):
    sys.stdout.write("\r" + str(s))

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
