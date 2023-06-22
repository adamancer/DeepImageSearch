import DeepImageSearch.config as config
import os
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torch
from torch.autograd import Variable
import timm
from PIL import ImageOps
import math
import faiss

if hash("test") != 4418353137104490830:
    raise ValueError("PYTHONHASHSEED must be set to 0")

class Load_Data:
    """A class for loading data from single/multiple folders or a CSV file"""

    def __init__(self):
        """
        Initializes an instance of LoadData class
        """
        pass
    
    def from_folder(self, folder_list: list, exts: tuple = ('.bmp', '.gif', '.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        """
        Adds images from the specified folders to the image_list.

        Parameters:
        -----------
        folder_list : list
            A list of paths to the folders containing images to be added to the image_list.
        """
        if isinstance(folder_list, str):
            folder_list = [folder_list]
        self.folder_list = folder_list
        image_path = []
        for folder in self.folder_list:
            for root, dirs, files in os.walk(folder):
                for fn in files:
                    if fn.lower().endswith(exts):
                        image_path.append(os.path.join(root, fn))
        return image_path

    def from_csv(self, csv_file_path: str, images_column_name: str):
        """
        Adds images from the specified column of a CSV file to the image_list.

        Parameters:
        -----------
        csv_file_path : str
            The path to the CSV file.
        images_column_name : str
            The name of the column containing the paths to the images to be added to the image_list.
        """
        self.csv_file_path = csv_file_path
        self.images_column_name = images_column_name
        return pd.read_csv(self.csv_file_path)[self.images_column_name].to_list()

class Search_Setup:
    """ A class for setting up and running image similarity search."""
    def __init__(self, image_list: list, model_name='vgg19', pretrained=True, image_count: int = None):
        """
        Parameters:
        -----------
        image_list : list
        A list of images to be indexed and searched.
        model_name : str, optional (default='vgg19')
        The name of the pre-trained model to use for feature extraction.
        pretrained : bool, optional (default=True)
        Whether to use the pre-trained weights for the chosen model.
        image_count : int, optional (default=None)
        The number of images to be indexed and searched. If None, all images in the image_list will be used.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.image_data = pd.DataFrame()
        self.d = None
        if image_count==None:
            self.image_list = image_list
        else:
            self.image_list = image_list[:image_count]

        if f'metadata-files/{self.model_name}' not in os.listdir():
            try:
                os.makedirs(f'metadata-files/{self.model_name}')
            except Exception as e:
                pass
                #print(f'\033[91m file already exists: metadata-files/{self.model_name}')

        # Load the pre-trained model and remove the last layer
        print("\033[91m Please Wait Model Is Loading or Downloading From Server!")
        base_model = timm.create_model(self.model_name, pretrained=self.pretrained)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval()
        print(f"\033[92m Model Loaded Successfully: {model_name}")

    def _extract(self, img):
        # Resize and convert the image
        img = img.resize((224, 224))
        img = img.convert('RGB')

        # Preprocess the image
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224, 0.225]),
        ])
        x = preprocess(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        # Extract features
        feature = self.model(x)
        feature = feature.data.numpy().flatten()
        return feature / np.linalg.norm(feature)

    def _get_feature(self, image_data: list):
        self.image_data = image_data
        features = []
        for img_path in tqdm(self.image_data):  # Iterate through images
            # Extract features from the image
            try:
                feature = self._extract(img=Image.open(img_path))
                features.append(feature)
            except:
               # If there is an error, append None to the feature list
               features.append(None)
               continue
        return features

    def _start_feature_extraction(self):
        image_data = pd.DataFrame()
        image_data['id'] = [hash(p) for p in self.image_list]
        image_data['images_paths'] = self.image_list
        f_data = self._get_feature(self.image_list)
        image_data['features'] = f_data
        image_data = image_data.dropna().reset_index(drop=True)
        image_data.to_pickle(config.image_data_with_features_pkl(self.model_name))
        print(f"\033[94m Image Meta Information Saved: [metadata-files/{self.model_name}/image_data_features.pkl]")
        return image_data

    def _start_indexing(self, image_data):
        self.image_data = image_data
        d = len(image_data['features'][0])  # Length of item vector that will be indexed
        self.d = d
        index = faiss.IndexIDMap(faiss.IndexFlatL2(d))
        features_matrix = np.vstack(image_data['features'].values).astype(np.float32)
        id_matrix = image_data['id']
        index.add_with_ids(features_matrix, id_matrix)  # Add the features matrix to the index
        faiss.write_index(index, config.image_features_vectors_idx(self.model_name))
        print("\033[94m Saved The Indexed File:" + f"[metadata-files/{self.model_name}/image_features_vectors.idx]")

    def run_index(self):
        """
        Indexes the images in the image_list and creates an index file for fast similarity search.
        """
        if len(os.listdir(f'metadata-files/{self.model_name}')) == 0:
            data = self._start_feature_extraction()
            self._start_indexing(data)
        else:
            print("\033[91m Metadata and Features are already present, Do you want Extract Again? Enter yes or no")
            flag = str(input())
            if flag.lower() == 'yes':
                data = self._start_feature_extraction()
                self._start_indexing(data)
            else:
                print("\033[93m Meta data already Present, Please Apply Search!")
                print(os.listdir(f'metadata-files/{self.model_name}'))
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name))
        self.f = len(self.image_data['features'][0])

    def add_images_to_index(self, new_image_paths: list):
        """
        Adds new images to the existing index.

        Parameters:
        -----------
        new_image_paths : list
            A list of paths to the new images to be added to the index.
        """
        # Load existing metadata and index
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name))
        index = faiss.read_index(config.image_features_vectors_idx(self.model_name))

        for new_image_path in tqdm(new_image_paths):
            # Extract features from the new image
            try:
                img = Image.open(new_image_path)
                feature = self._extract(img)
            except Exception as e:
                print(f"\033[91m Error extracting features from the new image: {e}")
                continue

            # Add the new image to the metadata
            img_id = hash(new_image_path)
            new_metadata = pd.DataFrame({"id": img_id, "images_paths": [new_image_path], "features": [feature]})
            #self.image_data = self.image_data.append(new_metadata, ignore_index=True)
            self.image_data = pd.concat([self.image_data, new_metadata], axis=0, ignore_index=True)

            # Add the new image to the index
            index.add_with_ids(np.array([feature], dtype=np.float32), [img_id])

        # Save the updated metadata and index
        self.image_data.to_pickle(config.image_data_with_features_pkl(self.model_name))
        faiss.write_index(index, config.image_features_vectors_idx(self.model_name))

        print(f"\033[92m New images added to the index: {len(new_image_paths)}")

    def remove_images_from_index(self, ids: list):
        """
        Removes new images from the existing index.

        Parameters:
        -----------
        ids : list
            A list of ids for the images to be removed. Currently these are hashes
            of the filepath with PYTHONHASHSEED=0.
        """
        image_data = self.get_image_metadata_file()
        index = self.get_image_index_file()
        
        image_data = image_data[~image_data.id.isin(ids)]
        index.remove_ids(faiss.IDSelectorBatch(ids))
        
        image_data.to_pickle(config.image_data_with_features_pkl(self.model_name))
        faiss.write_index(index, config.image_features_vectors_idx(self.model_name))

        self.image_data = image_data

    def _search_by_vector(self, v, n: int):
        self.v = v
        self.n = n
        index = faiss.read_index(config.image_features_vectors_idx(self.model_name))
        D, I = index.search(np.array([self.v], dtype=np.float32), self.n)
        matches = self.image_data[self.image_data['id'].isin(I[0])]['images_paths'].to_list()
        return dict(zip(I[0], matches))

    def _get_query_vector(self, image_path: str):
        self.image_path = image_path
        img = Image.open(self.image_path)
        query_vector = self._extract(img)
        return query_vector

    def plot_similar_images(self, image_path: str, number_of_images: int = 6):
        """
        Plots a given image and its most similar images according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image to be plotted.
        number_of_images : int, optional (default=6)
            The number of most similar images to the query image to be plotted.
        """
        img = Image.open(image_path)
        img.thumbnail((448, 448), Image.LANCZOS)
        
        query_vector = self._get_query_vector(image_path)
        img_list = list(self._search_by_vector(query_vector, number_of_images).values())
        row_size = 5
        rows = []
        while img_list:
            rows.append(img_list[:row_size])
            img_list = img_list[row_size:]

        fig = plt.figure(figsize=(4.48 * 3, 2 * len(rows)), dpi=100)
        outer = gridspec.GridSpec(1, 2, width_ratios=(1, 2), wspace=0.1)
        
        # Show search image
        main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0])
        ax = plt.Subplot(fig, main[0])
        ax.imshow(img)
        ax.set_title("Query Image")
        ax.axis('off')
        fig.add_subplot(ax)
        
        gallery = gridspec.GridSpecFromSubplotSpec(
            len(rows),
            len(rows[0]),
            subplot_spec=outer[1],
        )
        i = 0
        for row in rows:
            for path in row:
                ax = plt.Subplot(fig, gallery[i])
                img = Image.open(path)
                img.thumbnail((224, 224), Image.LANCZOS)
                ax.set_title(f"{i + 1}.")
                ax.imshow(img)
                ax.axis('off')
                fig.add_subplot(ax)
                i += 1
        
        fig.show()

    def get_similar_images(self, image_path: str, number_of_images: int = 10):
        """
        Returns the most similar images to a given query image according to the indexed image features.

        Parameters:
        -----------
        image_path : str
            The path to the query image.
        number_of_images : int, optional (default=10)
            The number of most similar images to the query image to be returned.
        """
        self.image_path = image_path
        self.number_of_images = number_of_images
        query_vector = self._get_query_vector(self.image_path)
        img_dict = self._search_by_vector(query_vector, self.number_of_images)
        return img_dict
    
    def get_image_metadata_file(self):
        """
        Returns the metadata file containing information about the indexed images.

        Returns:
        --------
        DataFrame
            The Panda DataFrame of the metadata file.
        """
        self.image_data = pd.read_pickle(config.image_data_with_features_pkl(self.model_name))
        return self.image_data
    
    def get_image_index_file(self):
        """
        Returns the metadata file containing information about the indexed images.

        Returns:
        --------
        DataFrame
            The Panda DataFrame of the metadata file.
        """
        return faiss.read_index(config.image_features_vectors_idx(self.model_name))
