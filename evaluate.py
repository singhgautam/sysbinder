import argparse

import torch

from torch.utils.data import DataLoader

import numpy as np

from data import CLEVREasyWithAnnotations
from sysbinder import SysBinderImageAutoEncoder
from hungarian import Hungarian
from dci import gbt, dci

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


def hungarian_align(true, pred):
    """
        Order for re-ordering the predicted masks
    :param true masks: B, N, 1, H, W
    :param pred masks: B, M, 1, H, W
    :return:
    """

    intersection = true[:, :, None, :, :, :] * pred[:, None, :, :, :, :]  # B, N, M, 1, H, W
    intersection = intersection.flatten(start_dim=3).sum(-1)  # B, N, M

    union = -intersection
    union += true[:, :, None, :, :, :].flatten(start_dim=3).sum(-1)
    union += pred[:, None, :, :, :, :].flatten(start_dim=3).sum(-1)

    iou = intersection / union

    orders = []
    for b in range(iou.shape[0]):
        profit_matrix = iou[b].cpu().numpy()  # N, M
        hungarian = Hungarian(profit_matrix, is_profit_matrix=True)
        hungarian.calculate()
        results = hungarian.get_results()
        order = [j for (i,j) in results]
        orders += [order]

    orders = torch.Tensor(orders).long().to(iou.device)  # B, N

    return orders


with torch.no_grad():

    representations = []
    ground_truth_factors = []

    print("Building points...")
    for batch, (image, masks, factors) in enumerate(test_loader):
        image = image.cuda()  # (B, C, H, W)
        masks = masks.cuda()  # (B, N0, 1, H, W)

        factors = factors.cuda()  # (B, N0, F)

        slots, _, masks_pred = model.encode(image)  # (B, N, D), _, (B, N, 1, H, W)

        order = hungarian_align(masks, masks_pred)  # B, N
        slots = torch.gather(slots, 1, order[:, :, None].expand(-1, -1, args.slot_size))  # B, N, D
        masks_pred = torch.gather(masks_pred, 1, order[:, :, None, None, None].expand(-1, -1, 1, args.image_size, args.image_size))  # B, N, 1, H, W

        representations += [slots.flatten(end_dim=1).cpu()]
        ground_truth_factors += [factors.flatten(end_dim=1).cpu()]

        if batch * args.batch_size >= args.num_points:
            break

    representations = torch.cat(representations, 0).numpy()  # B, D
    ground_truth_factors = torch.cat(ground_truth_factors, 0).numpy()  # B, G

    torch.cuda.empty_cache()

    print("Points built.")

    importance_matrix, _, informativeness = gbt(representations.T, ground_truth_factors.T)  # [num_codes, num_factors]
    num_codes, num_factors = importance_matrix.shape

    importance_matrix = np.reshape(importance_matrix, (args.num_blocks, args.slot_size // args.num_blocks, num_factors))  # [M, d, G]
    importance_matrix = importance_matrix.mean(axis=1)  # [M, G]

    disentanglement, completeness = dci(importance_matrix)

    print(f'DCI Disentanglement = {disentanglement} \t Completeness = {completeness} \t Informativeness = {informativeness}')
