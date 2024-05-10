
import os
from PIL import Image
import numpy as np
import lmdb
import pandas as pd
import pickle
import tqdm
import torch
torch.manual_seed(123456)


class LMDB_Image:
    def __init__(self, image, id):
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.id = id

    def get_image(self):
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)


if __name__ == '__main__':
    name_list = ['Industrial_and_Scientific', 'Musical_Instruments', 'Office_Products']
    folder_name_list = ['Scientific', 'Instrument', 'Office']



    for index, name in enumerate(name_list):
        print(f"processing {name}")
        print('build lmdb database')
        nc_items = pd.read_table(f'{folder_name_list[index]}/{name}_items.tsv', header=None)
        nc_items.columns=['item_id',"title"]
        nc_items = nc_items[['item_id']]
        image_num = len(nc_items)
        print("all images %s" % image_num)

        lmdb_path = f'am_{"".join([name.split("_")[i][0].lower() for i in [0,-1]])}.lmdb'
        isdir = os.path.isdir(lmdb_path)
        print("Generate LMDB to %s" % lmdb_path)
        lmdb_env = lmdb.open(lmdb_path, subdir=isdir, map_size=image_num * np.zeros((3, 224, 224)).nbytes,
                             readonly=False, meminit=False, map_async=True)
        txn = lmdb_env.begin(write=True)
        write_frequency = 5000

        image_file = f'am_image_{"".join([name.split("_")[i][0].lower() for i in [0,-1]])}'
        bad_file = {}

        lmdb_keys = []
        for index, row in tqdm.tqdm(nc_items.iterrows()):
            item_id = row[0]
            item_path = item_id + '.jpg'

            lmdb_keys.append(item_id)
            img = np.array(Image.open(os.path.join(image_file, item_path)).convert('RGB'))
            temp = LMDB_Image(img, item_id)
            txn.put(u'{}'.format(item_id).encode('ascii'), pickle.dumps(temp))
            if index % write_frequency == 0 and index != 0:
                txn.commit()
                txn = lmdb_env.begin(write=True)
            try:
                img = np.array(Image.open(os.path.join(image_file, item_path)).convert('RGB'))
                temp = LMDB_Image(img, item_id)
                txn.put(u'{}'.format(item_id).encode('ascii'), pickle.dumps(temp))
                if index % write_frequency == 0 and index != 0:
                    txn.commit()
                    txn = lmdb_env.begin(write=True)
            except Exception as e:
                print('bad',item_id)
                bad_file[index] = item_id

        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in lmdb_keys]
        with lmdb_env.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
            txn.put(b'__len__', pickle.dumps(len(keys)))
        print(len(keys))
        print("Flushing database ...")
        lmdb_env.sync()
        lmdb_env.close()

        print('bad_file  ', len(bad_file))
        bad_url_df = pd.DataFrame.from_dict(bad_file, orient='index', columns=['item_id'])
        bad_url_df.to_csv('lmdb_bad_file.tsv', sep='\t', header=None, index=False)


