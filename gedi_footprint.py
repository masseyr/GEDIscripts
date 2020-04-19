from geosoup import Vector, Handler, Opt
from scipy.stats import binned_statistic
import multiprocessing as mp
from osgeo import ogr
import numpy as np
import itertools
import operator
import h5py
import json
import sys


'''
Script to draw extents of GEDI products inside a directory. 
The output is a shapefile with Geographic projection and 
WGS84 datum. the attributes of the shapefile include:
BEAM, YEAR, JDAY, FILE

usage:
python gedi_footprint.py [gedi directory] [output shapefile] [number of parallel processes]
'''


def get_path(filename,
             res=0.5,
             buffer=0.000135):
    """
    Method to extract path from a GEDI file
    :param filename: GEDI filename
    :param res: bin resolution (degrees)
    :param buffer: buffer from outermost point (degrees)
    :return: (attribute dictionary, geometry WKT, None) if no error is raised while opening file
            (None, None, error string) if error is raised
    """
    date_str = Handler(filename).basename.split('_')[2]

    year = int(date_str[0:4])
    julian_day = int(date_str[4:7])

    bin_edges = np.arange(-180.0, 180, res)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    if bin_edges[-1] < 180.0:
        bin_edges = np.hstack([bin_edges, np.array([180.0])])

    x_coords = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if buffer < res:  # if buffer is more than or eq to half bin width, ignore buffer
        x_coords[0] = x_coords[0] + buffer
        x_coords[-1] = x_coords[-1] - buffer

    file_keys = []
    try:
        fs = h5py.File(filename, 'r')
        fs.visit(file_keys.append)
    except Exception as e:
        return Handler(filename).basename, ' '.join(e.args)

    beam_ids = list(set(list(key.split('/')[0].strip() for key in file_keys if 'BEAM' in key)))

    feat_list = []
    err = 'No Keys found'

    for beam in beam_ids:

        beam_id = int(beam.replace('BEAM', ''), 2)

        try:
            lat_arr = np.array(fs['{}/geolocation/latitude_bin0'.format(beam)])
            lon_arr = np.array(fs['{}/geolocation/longitude_bin0'.format(beam)])
        except Exception as e:
            err = ' '.join(e.args)
            continue

        pos_arr = np.vstack([lon_arr, lat_arr]).T

        nan_loc_pre = np.where(np.apply_along_axis(lambda x: not (np.isnan(x[0]) or np.isnan(x[1])), 1, pos_arr))
        pos_arr = pos_arr[nan_loc_pre]

        pos_arr = pos_arr[np.lexsort((pos_arr[:, 0], pos_arr[:, 1]))]

        upper_lims, _, _ = binned_statistic(pos_arr[:, 0], pos_arr[:, 1], statistic='max', bins=bin_edges)
        lower_lims, _, _ = binned_statistic(pos_arr[:, 0], pos_arr[:, 1], statistic='min', bins=bin_edges)

        nan_loc = np.where(np.isnan(upper_lims))
        nan_groups = group_consecutive(nan_loc[0].tolist())

        chunks = locate_slice_by_group(nan_groups, x_coords.shape[0])

        main_geom = ogr.Geometry(ogr.wkbMultiPolygon)

        for chunk in chunks:
            if chunk[0] >= chunk[1]:
                continue
            else:
                chunk_bin_widths = bin_widths[chunk[0]:chunk[1]]

                upper_bounds = np.vstack([x_coords[chunk[0]:chunk[1]],
                                          upper_lims[chunk[0]:chunk[1]]])

                lower_bounds = np.vstack([x_coords[chunk[0]:chunk[1]],
                                          lower_lims[chunk[0]:chunk[1]]])

                corr_upper_bounds, corr_lower_bounds = correct_bounds(upper_bounds,
                                                                      lower_bounds,
                                                                      chunk_bin_widths)

                corr_lower_bounds = np.flip(corr_lower_bounds, 1)

                part_geom_coords = np.hstack([corr_upper_bounds, corr_lower_bounds]).T
                part_geom_coords = np.vstack([part_geom_coords, part_geom_coords[0, :]])

                part_geom_json = json.dumps({'type': 'Polygon', 'coordinates': [part_geom_coords.tolist()]})
                part_geom = Vector.get_osgeo_geom(part_geom_json, 'json')

                # out_geom = geom.Buffer(buffer)
                part_geom.Buffer(buffer)

                # main_geom.AddGeometryDirectly(out_geom)
                main_geom.AddGeometryDirectly(part_geom)

        attributes = {'BEAM': beam_id,
                      'FILE': Handler(filename).basename,
                      'YEAR': year,
                      'JDAY': julian_day}

        feat_list.append((main_geom.ExportToWkt(), attributes))

    if len(feat_list) == 0:
        return Handler(filename).basename, err
    else:
        return feat_list, None


def correct_bounds(u_bounds,
                   l_bounds,
                   bin_widths):
    """
    Method to correct upper and lower bounds of GEDI beam path
    The correction is most needed when he path is least parallel to equator and vice versa
    :param u_bounds: upper bounds of beam path : array([[x1, x2, ....],
                                                        [y1, y2, ....]])
    :param l_bounds: lower bounds of beam path : array([[x1, x2, ....],
                                                        [y1, y2, ....]])
    :param bin_widths: bin widths of type array([b1, b2, ...])
    :return: tuple of (corrected upper bounds, corrected lower bounds)
    """

    if u_bounds.shape[1] <= 1:
        return u_bounds, l_bounds

    else:
        y_diff = (u_bounds[1, :] - l_bounds[1, :]) / 2.0
        x_diff = bin_widths / 2.0

        centr_coords = (u_bounds + l_bounds) / 2.0

        theta = np.arctan((u_bounds[1, :-1] - u_bounds[1, 1:]) /
                          (u_bounds[0, :-1] - u_bounds[0, 1:]))

        theta = np.hstack([theta, np.array([theta[-1]])])

        tan_theta = np.abs(np.tan(theta))
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        y_corr = (y_diff - (x_diff * tan_theta)) * (cos_theta ** 2)
        x_corr = (y_diff - (x_diff * tan_theta)) * (cos_theta * sin_theta)

        corr = np.vstack([x_corr,
                          y_corr])

        new_u_bounds = centr_coords + corr
        new_l_bounds = centr_coords - corr

        return new_u_bounds, new_l_bounds


def group_consecutive(arr):
    """
    Method to group consecutive numbers
    :param arr: Array or list of numbers
    :return: list of lists of grouped numbers
    """
    grouped = []
    for _, group in itertools.groupby(enumerate(sorted(arr)), key=lambda x: x[0] - x[1]):
        grouped.append(list(map(operator.itemgetter(1), group)))
    return grouped


def locate_slice_by_group(pts, length):
    """
    Methods to slice an array using grouped discontinuity output from group_consecutive()
    :param pts: list of list of grouped discontinuities
    :param length: Initial length of array
    :return: list of tuples of start and end location of slices
    """
    if len(pts) == 0:
        return [[0, length - 1]]
    else:
        slices = []
        next_pt = None
        for pt in pts:
            if next_pt is None:
                slices.append([0, pt[0]])
                next_pt = pt[-1]
            else:
                slices.append([next_pt + 1, pt[0]])
                next_pt = pt[-1]

        if (next_pt + 1) < length:
            slices.append([next_pt + 1, length - 1])

        if slices[0] == [0, 0]:
            return slices[1:]
        else:
            return slices


if __name__ == '__main__':

    script, gedi_dir, outfile, nproc = sys.argv

    nproc = int(nproc)

    attrib = {'BEAM': 'int', 'FILE': 'str', 'YEAR': 'int', 'JDAY': 'int'}

    vec = Vector(name='gedi_extent',
                 epsg=4326,
                 geom_type='multipolygon',
                 filename=outfile,
                 attr_def=attrib)

    file_list = Handler(dirname=gedi_dir).find_all('*.h5')

    with mp.Pool(processes=nproc) as pool:

        for file_output, err_str in pool.imap_unordered(get_path, file_list):
            if err_str is None and len(file_output) > 0:
                attr_list = []
                for geom_wkt, attrs in file_output:
                    geom = ogr.CreateGeometryFromWkt(geom_wkt)
                    attr_list.append(attrs)
                    vec.add_feat(geom, attr=attrs)

                Opt.cprint(str(list(set([attr['FILE'] for attr in attr_list]))[0]) + ' : Processed')
            else:
                Opt.cprint(file_output, newline=' : ')
                Opt.cprint(err_str)

    vec.datasource = None

    Opt.cprint('\n----------------------------------------------------------')
    Opt.cprint(vec)
    Opt.cprint(outfile)

