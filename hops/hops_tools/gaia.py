
import sqlite3
import os
import time
import traceback

import numpy as np
from astropy.table import Table


def separation(ra1, dec1, ra2, dec2):
    return (180.0/np.pi) * np.arccos(np.minimum(1, np.sin(dec1* np.pi / 180.0) * np.sin(dec2 * np.pi / 180.0) +
                                                np.cos(dec1* np.pi / 180.0) * np.cos(dec2 * np.pi / 180.0) * np.cos(ra1* np.pi / 180.0 - ra2* np.pi / 180.0)))


def cos_separation(ra1, dec1, ra2, cos_dec2, sin_dec2):

    xx = np.sin(dec1* np.pi / 180.0) * sin_dec2 + np.cos(dec1* np.pi / 180.0) * cos_dec2 * np.cos(ra1* np.pi / 180.0 - ra2* np.pi / 180.0)

    if xx:
        return xx
    else:
        return -1


class LocalGAIA:

    def __init__(self, local_database_path):

        self.local_database_path = local_database_path
        self.con = None
        self.cur = None

        try:
            self.con = sqlite3.connect(os.path.join(self.local_database_path, "gaia.db"))
            self.con.create_function('separation', 4, separation)
            self.cur = self.con.cursor()
        except Exception as e:
            print('Cannot use local GAIA DB.')
            print(f'{traceback.format_exc()}')

    def get_sectors(self, ra, dec, radius):

        if self.cur is None:

            return None

        query = "SELECT id, "
        query += "separation(ra, dec, {0}, {1}) as dist ".format(ra, dec)
        query += "FROM sectors WHERE dist < radius + {0} ".format(radius)

        q = self.cur.execute(query).fetchall()

        if len(q) == 0:
            query = "SELECT id, "
            query += "separation(ra, dec, {0}, {1}) as dist ".format(ra, dec)
            query += "FROM sectors ORDER BY dist"

            q = self.cur.execute(query).fetchone()

        sectors = []
        for sector_id in np.array(q, dtype=int)[:, 0]:

            conn = sqlite3.connect(os.path.join(self.local_database_path, "{0}.db".format(sector_id)))
            conn.create_function('separation', 4, separation)
            conn.create_function('cos_separation', 5, cos_separation)

            sectors.append([conn, sector_id])

        return sectors

    def fix_sector_file(self, sector_id):

        print("Fixing sector {0}".format(sector_id))
        con = sqlite3.connect(os.path.join(self.local_database_path, "gaia.db"))
        with con:
            box = con.execute("SELECT * FROM sectors WHERE id=={0}".format(sector_id)).fetchall()[0]
        con.close()

        query = "SELECT TOP {0} source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec ".format(10000000)
        query += "FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), BOX('ICRS', {0}, {1}, {2}, {3}))".format(box[1], box[2], 1.05*abs(box[4]-box[3]), 1.05*abs(box[6]-box[5]))
        job = Gaia.launch_job_async(query)

        q = job.get_results()

        data = [[int(q['SOURCE_ID'][xx]),
                 float(q['ra'][xx]),
                 float(q['dec'][xx]),
                 float(q['phot_g_mean_mag'][xx]),
                 float(q['phot_bp_mean_mag'][xx]),
                 float(q['phot_rp_mean_mag'][xx]),
                 float(q['parallax'][xx]),
                 float(q['pmra'][xx]),
                 float(q['pmdec'][xx]),
                 ] for xx in range(len(q))]

        sector_path = os.path.join(local_database_path, "{0}.db".format(sector_id))

        if os.path.isfile(sector_path):
            os.remove(sector_path)

        sector_con = sqlite3.connect(sector_path)
        with sector_con:
            sector_con.execute("CREATE TABLE sources(source_id int primary key, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec)")
            sector_con.executemany("INSERT OR IGNORE INTO sources VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
            sector_con.commit()
        sector_con.close()

    def get_gaia_stars(self, ra_0, dec_0, radius, limit):

        if self.cur is not None:
            results = []

            sectors = self.get_sectors(ra_0, dec_0, radius)

            for sector in sectors:
                query = "SELECT source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec,"
                query += "cos_separation(ra, dec, {1}, {2}, {3}) as dist FROM sources WHERE phot_g_mean_mag > 1 AND dist > {4} ORDER BY phot_g_mean_mag ASC LIMIT {0}".format(limit, ra_0, np.cos(dec_0*np.pi/180), np.sin(dec_0*np.pi/180), np.cos(radius*np.pi/180))
                for trial in range(3):
                    try:
                        try:
                            results += sector[0].execute(query).fetchall()
                            break

                        except sqlite3.DatabaseError:
                            sector[0].close()
                            fix_sector_file(sector[1])
                            sector[0] = sqlite3.connect(os.path.join(self.local_database_path, "{0}.db".format(sector[1])))
                            sector[0].create_function('separation', 4, separation)
                            sector[0].create_function('cos_separation', 5, cos_separation)

                            results += sector[0].execute(query).fetchall()
                            break

                    except sqlite3.OperationalError:
                        time.sleep(0.1)

            for sector in sectors:
                sector[0].close()

            del sectors

            results = sorted(results, key=lambda x: x[3])[:limit]

            data = Table()
            source_ids = []
            source_idxs = []
            for i in range(len(results)):
                if len(source_ids) >= limit:
                    break
                else:
                    if results[i][0] not in source_ids:
                        try:
                            _ = int(results[i][0])
                            source_ids.append(results[i][0])
                            source_idxs.append(i)
                        except ValueError:
                            pass

            data['SOURCE_ID'] = np.array([results[ff][0] for ff in source_idxs], dtype=np.int64)
            data['ra'] = np.array([results[ff][1] for ff in source_idxs], dtype=float)
            data['dec'] = np.array([results[ff][2] for ff in source_idxs], dtype=float)
            data['phot_g_mean_mag'] = np.array([results[ff][3] for ff in source_idxs], dtype=float)
            data['phot_bp_mean_mag'] = np.array([results[ff][4] for ff in source_idxs], dtype=float)
            data['phot_rp_mean_mag'] = np.array([results[ff][5] for ff in source_idxs], dtype=float)
            data['parallax'] = np.array([results[ff][6] for ff in source_idxs], dtype=float)
            data['pmra'] = np.array([results[ff][7] for ff in source_idxs], dtype=float)
            data['pmdec'] = np.array([results[ff][8] for ff in source_idxs], dtype=float)

            return data

        else:

            onlinegaia = OnlineGAIA()

            return onlinegaia.get_gaia_stars(ra_0, dec_0, radius, limit)


class OnlineGAIA:

    def __init__(self):
        pass


    def get_gaia_stars(self, ra_0, dec_0, radius, limit):

        from astroquery.gaia import Gaia

        Gaia.ROW_LIMIT = limit
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

        query = "SELECT TOP {0} {1} FROM {2} WHERE phot_g_mean_mag > 1 AND parallax > 0 AND 1 = CONTAINS(POINT('ICRS', {3}, {4}), " \
                "CIRCLE('ICRS', ra, dec, {5})) ORDER BY phot_g_mean_mag ASC".format(
            limit,
            'source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, parallax, pmra, pmdec',
            Gaia.MAIN_GAIA_TABLE,
            ra_0,
            dec_0,
            radius,
        )
        job = Gaia.launch_job_async(query)

        return job.get_results()
