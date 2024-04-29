from lightning.fabric.utilities import cloud_io

from eva.core.loggers.json_logger import JSONLogger

logger = JSONLogger(root_dir=".")

# print(logger)



root_dir = versions_root = "dummy"

fs = cloud_io.get_filesystem(root_dir)
# print(cloud_io._is_dir(fs, versions_root, strict=True))


# for directory in fs.listdir(versions_root):
#     print(directory)


strings = [
    "/Users/ioangatop/Desktop/dev2/eva/dummy/edwin",
    "/Users/ioangatop/Desktop/dev2/eva/dummy/version_1",
    "/Users/ioangatop/Desktop/dev2/eva/dummy/version_",
    "/Users/ioangatop/Desktop/dev2/eva/dummy/version_100",
    "/Users/ioangatop/Desktop/dev2/eva/dummy/version_1",
    "/Users/ioangatop/Desktop/dev2/eva/dummy/version_0",
    "/Users/ioangatop/Desktop/dev2/eva/version_0/dummy/afa",
    "/Users/ioangatop/Desktop/dev2/eva/dummy/version_5",
    "/Users/ioangatop/Desktop/dev2/eva/dummy/version_10",
]


import re


def extract_num(s, p, ret=0):
    search = p.search(s)
    if search:
        return int(search.groups()[0])
    else:
        return ret


# print re.findall(r"(?<=Version\s)\S+",x)

# filtered_list = list(filter(lambda s: not re.match(r'.*\d', s), strings))
# filtered_list = list(filter(lambda s: not re.match(r"(?<=version_\s)\S+", s), strings))
# filtered_list = list(filter(lambda s: not re.match(r"(?P<version_>\w+)", s), strings))

versioned_logs = list(filter(lambda s: re.findall(r"version_\d", s.split("/")[-1]), strings))
sorted_logs = sorted(versioned_logs, key=lambda s: int(re.findall(r'(\d+)', s)[-1]))
print(sorted_logs)


# print(sorted(x, key=lambda s: int(re.findall(r'(\d+)', s)[-1])))




# s = "/Users/ioangatop/Desktop/dev2/eva/dummy/version_100"
# # re.search("(nn)", s[::-1])
# # print(re.search(r'\d+', s).group()[::-1])
# print()

