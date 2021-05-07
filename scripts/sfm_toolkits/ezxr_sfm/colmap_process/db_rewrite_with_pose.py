import sqlite3
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import networkx as nx # 图, 相关算法
IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))

def read_geo(file_path_name):
    with open(file_path_name, 'r') as file:
        data = file.read().split('\n')
        # delete empty line
        if (len(data[-1]) < 1):
            data = data[0:-1]
        geo_dict = {}
        for item in data:
            element = item.split(' ')
            cam_name = element[0].split('/')[-1]
            geo_dict[cam_name] = np.array([float(element[1]), float(element[2]), float(element[3])])
        # print(geo_dict)
    return geo_dict

def cal_dist(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2

def delete_row_with_id(db, row_id):
    cursor = db.cursor()
    cursor.execute("delete from two_view_geometries where pair_id=?", (row_id,))
    db.commit()

def rewrite_db_with_pose(db_name, src_geo_name, dist_thres):
    geo_dict = read_geo(src_geo_name)

    db = COLMAPDatabase.connect(db_name)

    cursor = db.cursor()
    cursor.execute("select * from two_view_geometries")
    matches_db = cursor.fetchall()
    cursor.close()
    print('two_view_geometries number without threshold:', len(matches_db))

    cursor = db.cursor()
    cursor = db.execute("SELECT image_id, name FROM images;")
    img_dict = {}
    for row in cursor:
    	img_dict[row[0]] = row[1]
    cursor.close()
    print('img_dict number:', len(img_dict))

    invalid_match_list = []
    delete_cnt = 0

    match_graph_db = nx.Graph()
    for match in matches_db:
        img_a_id, img_b_id = pair_id_to_image_ids(match[0])
        if int(match[1]) >= 15:
            image_id1 = str(int(img_a_id))
            image_id2 = str(int(img_b_id))
            match_graph_db.add_node(image_id1)
            match_graph_db.add_node(image_id2)
            match_graph_db.add_edge(image_id1, image_id2)

            a_xyz = geo_dict[ img_dict[img_a_id] ]
            b_xyz = geo_dict[ img_dict[img_b_id] ]
            dis = cal_dist(a_xyz, b_xyz)

            if (dis > dist_thres):
                delete_row_with_id(db, match[0])
                print ('delete_cnt:', delete_cnt, 'dis =', dis, ',', img_dict[img_a_id], ',', img_dict[img_b_id])
                delete_cnt += 1

    cursor.close()
    print('node number for all image = ', len(match_graph_db.nodes()))
    print('edge number for final match = ', len(match_graph_db.edges()))

    # for node in match_graph_db.nodes():
    #     if match_graph_db.degree[node] > 100:
    #         print('node = ', node, ' match number = ', match_graph_db.degree[node])

    print ()
    print('Total cnt:', len(match_graph_db.edges()), ', delete cnt:', delete_cnt, ', useful cnt', len(match_graph_db.edges()) - delete_cnt)

    db.close()

def view_db_match(db_name):
    db = COLMAPDatabase.connect(db_name)

    cursor = db.cursor()
    cursor.execute("select * from matches")
    matches_db = cursor.fetchall()
    cursor.close()
    print('matches number:', len(matches_db))

    db.close()

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

def view_db_descriptor(db_name):
    db = COLMAPDatabase.connect(db_name)

    cursor = db.cursor()
    cursor.execute("select * from descriptors")
    descriptors_db = cursor.fetchall()
    cursor.close()
    print('descriptors number:', len(descriptors_db))
    max_list = []
    min_list = []
    norm_list = []
    for id, descriptors in enumerate(descriptors_db):
        # print (descriptors[0])
        # print (descriptors[1])
        # print (descriptors[2])
        params = blob_to_array(descriptors[3], np.uint8)
        params = params.astype(np.int)
        norm = np.linalg.norm(params)

        max_list.append(np.max(params))
        min_list.append(np.min(params))
        norm_list.append(norm)
    plt.hist(max_list)
    plt.savefig('max.png')
    plt.close('all')
    plt.hist(min_list)
    plt.savefig('min.png')
    plt.close('all')
    plt.hist(norm_list)
    plt.savefig('norm.png')
    plt.close('all')
    db.close()

if __name__ == "__main__":
    
    db_name = '/home/fjt/Code/HF-Net/benchmark/ibl_dataset_cvpr17_3852/aslfeat_src.db'
    src_geo_name = '/home/fjt/Code/HF-Net/benchmark/ibl_dataset_cvpr17_3852/evo_eval/gt/geos.txt'
    dist_thres = 10.0

    rewrite_db_with_pose(db_name, src_geo_name, dist_thres)

    view_db_match(db_name)

    view_db_descriptor(db_name)


