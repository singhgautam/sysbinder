import argparse

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from torch.utils.data import DataLoader

from tqdm import tqdm

from data import CLEVREasyWithAnnotations
from sysbinder import SysBinderImageAutoEncoder

from kmeans_pytorch import kmeans


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--image_channels', type=int, default=3)
parser.add_argument('--num_points', type=int, default=2048)

parser.add_argument('--load_path', default='best_model.pt')
parser.add_argument('--data_path', default='test_set/CLEVR_new_??????????.png')

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=3)
parser.add_argument('--num_blocks', type=int, default=8)
parser.add_argument('--cnn_hidden_size', type=int, default=512)
parser.add_argument('--slot_size', type=int, default=2048)
parser.add_argument('--mlp_hidden_size', type=int, default=192)
parser.add_argument('--num_prototypes', type=int, default=64)

parser.add_argument('--vocab_size', type=int, default=4096)
parser.add_argument('--num_decoder_layers', type=int, default=8)
parser.add_argument('--num_decoder_heads', type=int, default=4)
parser.add_argument('--d_model', type=int, default=192)
parser.add_argument('--dropout', type=int, default=0.1)

args = parser.parse_args()

torch.manual_seed(args.seed)

test_dataset = CLEVREasyWithAnnotations(root=args.data_path, phase='full', img_size=args.image_size)
val_sampler = None

loader_kwargs = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': args.num_workers,
    'pin_memory': True,
    'drop_last': True,
}

test_loader = DataLoader(test_dataset, sampler=val_sampler, **loader_kwargs)

val_epoch_size = len(test_loader)

model = SysBinderImageAutoEncoder(args)
checkpoint = torch.load(args.load_path, map_location='cpu')
model.load_state_dict(checkpoint)
model = model.cuda()

with torch.no_grad():
    model.eval()
    block_size = args.slot_size // args.num_blocks
    result_path = 'clusters.png'

    NUM_CLUSTERS = 12
    NUM_SHOW_PER_CLUSTER = 80

    slots = []
    attns_vis = []

    for batch, (image, masks, factors) in enumerate(test_loader):

        image = image.cuda()
        B, C, H, W = image.shape

        _slots, _attns_vis, _attns = model.encode(image)
        _slots, _attns_vis, _attns = [x.flatten(end_dim=1) for x in (_slots, _attns_vis, _attns)]

        slots += [_slots.detach()]
        attns_vis += [_attns_vis.detach()]

        if batch == 4:
            break

    slots = torch.cat(slots, 0)
    attns_vis = torch.cat(attns_vis, 0)

    grids = []
    for block_index in tqdm(range(args.num_blocks)):

        blocks = slots[..., block_index * block_size: (block_index + 1) * block_size]
        assigned_ids, _ = kmeans(X=blocks, num_clusters=NUM_CLUSTERS, distance='cosine', device=blocks.device)

        grids_attribute = []
        for id in range(NUM_CLUSTERS):
            attns_vis_id = attns_vis[assigned_ids == id]
            grid_attribute_id = torch.zeros(NUM_SHOW_PER_CLUSTER, C, H, W).to(attns_vis_id.device)
            num_tiles = min(NUM_SHOW_PER_CLUSTER, attns_vis_id.shape[0])
            grid_attribute_id[:num_tiles] = attns_vis_id[:num_tiles]
            grid_attribute_id = vutils.make_grid(grid_attribute_id, nrow=10, pad_value=0.2)
            grids_attribute += [F.pad(grid_attribute_id, (16, 16, 16, 16), value=1.0)]
        grids += [torch.cat(grids_attribute, -2)]

    vutils.save_image(torch.cat(grids, -1), result_path)
    print(f"Saved at {result_path}")
