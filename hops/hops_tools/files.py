
import os
import ssl
import copy
import glob
import yaml
import numpy as np
import pickle
import urllib

from astropy.io import fits as pf
from urllib.request import urlretrieve


def find_fits_files(name_identifier):

    if len(name_identifier) > 0:

        fits_files_names = glob.glob('*{0}*.f*t*'.format(name_identifier)) + glob.glob('*{0}*.F*T*'.format(name_identifier))
        fits_files_names = list(np.unique(fits_files_names))
        fits_files_names.sort()
        return fits_files_names

    else:
        return []


def copy_fits(hdulist):
    return pf.HDUList([ff.copy() for ff in hdulist])


def open_fits(path):
    with pf.open(path, memmap=False) as hdulist:
        internal_copy = copy_fits(hdulist)
        try:
            internal_copy.verify('fix')
        except pf.VerifyError as e:
            hdu = int(str(e.args)[4:-4].split('\\n')[1].replace('HDU ', '').replace(':', ''))
            card = int(str(e.args)[4:-4].split('\\n')[2].replace('Card ', '').replace(':', ''))
            del internal_copy[hdu].header[card]
            internal_copy.verify('fix')
        for i in internal_copy:
            while i.header[-1] == '':
                i.header = i.header[:-1]
    return internal_copy


def save_fits(hdulist, path, overwrite=False):
    internal_copy = copy_fits(hdulist)
    try:
        internal_copy.writeto(path, output_verify='fix', overwrite=overwrite)
    except pf.VerifyError as e:
        hdu = int(str(e.args)[4:-4].split('\\n')[1].replace('HDU ', '').replace(':', ''))
        card = int(str(e.args)[4:-4].split('\\n')[2].replace('Card ', '').replace(':', ''))
        del internal_copy[hdu].header[card]
        internal_copy.writeto(path, output_verify='fix', overwrite=overwrite)

    del internal_copy


def get_fits_data(fits_file_name):

    with open_fits(fits_file_name) as fits_file:

        try:
            fits_data = [fits_file['SCI']]
        except KeyError:
            sci_id = 0
            for sci_id in range(len(fits_file)):
                try:
                    if fits_file[sci_id].data.all():
                        break
                except:
                    pass
            fits_data = [fits_file[sci_id]]

    return fits_data


def get_fits_data_and_header(path, return_hdu=False):

    with open_fits(path) as fits_file:

        try:
            fits_data = np.array(fits_file['SCI'].data)
            fits_header = {ff: fits_file['SCI'].header[ff] for ff in fits_file['SCI'].header}
        except KeyError:
            for idx in range(len(fits_file)):
                try:
                    fits_data = np.array(fits_file[idx].data,  dtype=float)
                    fits_header = {ff: fits_file[idx].header[ff] for ff in fits_file[idx].header if ff not in ['HISTORY', 'COMMENT', '']}
                    if isinstance(fits_file[idx].data, type(None)):
                        pass
                    else:
                        break
                except:
                    pass

        # bit10_test = np.sum((fits_data/64.0-np.int_(fits_data/64.0))**2) == 0
        # bit12_test = np.sum((fits_data/16.0-np.int_(fits_data/16.0))**2) == 0
        # bit14_test = np.sum((fits_data/4.0-np.int_(fits_data/4.0))**2) == 0
        #
        # if bit10_test:
        #     fits_data = fits_data/64.0
        #     fits_header['BITPIX'] = 10
        # elif bit12_test:
        #     fits_data = fits_data/16.0
        #     fits_header['BITPIX'] = 12
        # elif bit14_test:
        #     fits_data = fits_data/4.0
        #     fits_header['BITPIX'] = 14

        fits_data[np.where(np.isnan(fits_data))] = 1
        fits_data[np.where(fits_data == 0)] = 1

        if return_hdu:
            return fits_data, fits_header, fits_file

        else:
            return fits_data, fits_header


def open_dict(path):
    class Dummy(object):
        def __init__(self, *argv, **kwargs):
            pass

    _ = Dummy(5)

    class Unpickler(pickle._Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception as e:
                print(e)
                return Dummy

    with open(path, 'rb') as f:
        unpickler = Unpickler(f)
        return unpickler.load()


def save_dict(dictionary, path):
    internal_copy = copy_dict(dictionary)
    pickle.dump(internal_copy, open(path, 'wb'), protocol=2)
    del internal_copy


def copy_dict(dictionary):
    return copy.deepcopy(dictionary)


def open_yaml(path):
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def save_yaml(dictionary, path):
    yaml.dump(dictionary, open(path, 'w'), default_flow_style=False)


def copy_yaml(dictionary):
    return copy.deepcopy(dictionary)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def download(link, destination, filename=None, verbose=True):
    if not filename:
        filename = os.path.split(destination)[1]

    if verbose:
        print('    Downloading {0}...'.format(filename))
        print('           from {0} '.format(link))
    try:
        with urllib.request.urlopen(link, context=ctx) as u, \
                open(destination, 'wb') as f:
            f.write(u.read())
        if verbose:
            print('    Done!')
        return True
    except Exception as e:
        print('ERROR: {0}\n    Could not download {1}', e, filename)
        return False


def open_dict_online(link):
    try:
        return pickle.load(urllib.request.urlopen(link, context=ctx))
    except Exception as e:
        print('Could not open dictionary {0}: {1}'.format(link, e))
        return False

