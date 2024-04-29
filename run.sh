EMBEDDINGS_ROOT="./data/embeddings/dino_vits16_lunit" \                                      
CHECKPOINT_PATH="https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/dino_vit_small_patch16_ep200.torch" \
NORMALIZE_MEAN=[0.70322989,0.53606487,0.66096631] \
NORMALIZE_STD=[0.21716536,0.26081574,0.20723464] \
eva predict_fit --config configs/vision/dino_vit/offline/bach.yaml
