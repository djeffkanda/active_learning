"""
University of Sherbrooke
NN Class Project
Authors: D'Jeff Kanda, Gabriel McCarthy, Mohamed Ragued
"""
import os
import tarfile
from urllib.request import urlretrieve
#
# def make_dir(dir_name):
#     """
#     Create a directory safely
#     Args:
#         directory name
#     """
#     current_dir = os.getcwd()
#     try:
#         os.mkdir(os.path.join(current_dir, dir_name))
#     except OSError:
#         print("Failed to create {}".format(dir_name))
#
#
# def download(download_url, local_destination, expected_bytes=None):
#     """
#     Download a file from download_url into local_destination if
#     the file doesn't already exists.
#     if expected_bytes is provided check if the downloaded file has the same
#     number of bytes.
#     """
#     if os.path.exists(local_destination):
#         print("{} already exists".format(local_destination))
#     else:
#         print("Downloading {}...".format(download_url))
#         local_file, headers = urlretrieve(download_url, local_destination)
#         file_stat = os.stat(local_destination)
#         if expected_bytes:
#             if file_stat.st_size == expected_bytes:
#                 print("Successfully downloaded {}".format(local_destination))
#
#
# # def download_and_extract_cifar10():
# #     """
# #     Download and extract cifar10 dataset. You can either use this function or
# #     use the API provided by pytorch.
# #     """
# #     cifar_dir = os.path.join(os.getcwd(), "data")
# #     make_dir("data")
# #     local_file = os.path.join(cifar_dir, "tmp_file")
# #     download(CIFAR_URL, local_file, CIFAR_SIZE)
# #     tf = tarfile.open(local_file)
# #     tf.extractall(path=cifar_dir)
# #     os.remove(local_file)
# #     tf.close()
