id2label = {
    0: "background",
    1: "candy",
    2: "egg tart",
    3: "french fries",
    4: "chocolate",
    5: "biscuit",
    6: "popcorn",
    7: "pudding",
    8: "ice cream",
    9: "cheese butter",
    10: "cake",
    11: "wine",
    12: "milkshake",
    13: "coffee",
    14: "juice",
    15: "milk",
    16: "tea",
    17: "almond",
    18: "red beans",
    19: "cashew",
    20: "dried cranberries",
    21: "soy",
    22: "walnut",
    23: "peanut",
    24: "egg",
    25: "apple",
    26: "date",
    27: "apricot",
    28: "avocado",
    29: "banana",
    30: "strawberry",
    31: "cherry",
    32: "blueberry",
    33: "raspberry",
    34: "mango",
    35: "olives",
    36: "peach",
    37: "lemon",
    38: "pear",
    39: "fig",
    40: "pineapple",
    41: "grape",
    42: "kiwi",
    43: "melon",
    44: "orange",
    45: "watermelon",
    46: "steak",
    47: "pork",
    48: "chicken duck",
    49: "sausage",
    50: "fried meat",
    51: "lamb",
    52: "sauce",
    53: "crab",
    54: "fish",
    55: "shellfish",
    56: "shrimp",
    57: "soup",
    58: "bread",
    59: "corn",
    60: "hamburg",
    61: "pizza",
    62: "hanamaki baozi",
    63: "wonton dumplings",
    64: "pasta",
    65: "noodles",
    66: "rice",
    67: "pie",
    68: "tofu",
    69: "eggplant",
    70: "potato",
    71: "garlic",
    72: "cauliflower",
    73: "tomato",
    74: "kelp",
    75: "seaweed",
    76: "spring onion",
    77: "rape",
    78: "ginger",
    79: "okra",
    80: "lettuce",
    81: "pumpkin",
    82: "cucumber",
    83: "white radish",
    84: "carrot",
    85: "asparagus",
    86: "bamboo shoots",
    87: "broccoli",
    88: "celery stick",
    89: "cilantro mint",
    90: "snow peas",
    91: "cabbage",
    92: "bean sprouts",
    93: "onion",
    94: "pepper",
    95: "green beans",
    96: "French beans",
    97: "king oyster mushroom",
    98: "shiitake",
    99: "enoki mushroom",
    100: "oyster mushroom",
    101: "white button mushroom",
    102: "salad",
    103: "other ingredients"
}


from transformers import Dinov2Model, Dinov2PreTrainedModel

# model = Dinov2Model(config)

model = Dinov2Model.from_pretrained(
    "facebook/dinov2-base",
    id2label=id2label,
    num_labels=len(id2label),
)

import torch
from eva.vision.utils.io import read_image

image = read_image("image.png")
mask = read_image("mask.png")

pixel_values = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
pixel_values = pixel_values / 255.

outputs = model(
    pixel_values,
    output_hidden_states=False,
    output_attentions=False,
)
print(outputs.last_hidden_state.shape)  # torch.Size([1, 973, 768])
# get the patch embeddings - so we exclude the CLS token
patch_embeddings = outputs.last_hidden_state[:,1:,:]
print(patch_embeddings.last_hidden_state.shape)  # torch.Size([1, 972, 768])


import timm


# vision transformer outputs "patch embeddings", meaning an embedding vector for each image patch
# model outputs a tensor of shape (batch_size, number of patches, hidden_size) = (batch_size, 1024, 768)
# (the model has a hidden size - also called embedding dimension - of 768 as seen here)

# backbone = timm.create_model(
#     "vit_small_patch16_224",
#     pretrained=False,
#     features_only=True,
#     # dynamic_img_size=True,
#     out_indices=(-1,),
# )
# pixel_values = torch.Tensor(2, 3, 224, 224)
# outputs = backbone(pixel_values)

# last_hidden_state = outputs[0]
# last_hidden_state = last_hidden_state.reshape(last_hidden_state.shape[0], last_hidden_state.shape[1], -1)
# print(outputs[-1].shape)
# print(last_hidden_state.shape)



# backbone = timm.create_model("vit_small_patch16_224")
# pixel_values = torch.Tensor(2, 3, 224, 224)
# outputs = backbone(pixel_values)

# out = backbone.get_intermediate_layers(pixel_values, n=1)
# out = out[-1]
# out = out[:, 1:, :]  # we discard the [CLS] token

# print(out.shape)
# # [0]
# # 
# # print(out.shape)
# #  dim x h*w
