import torch


# Create blend masks
def create_lin_mask(overlap, special_overlap, is_special, d, d2, rotate, c, device):
    mask_tensors = []
    if is_special:
        ones_size = special_overlap
        mask_tensors += [torch.zeros(d - (special_overlap + overlap), device=device)]
    else:
        ones_size = d - overlap
    mask_tensors += [torch.linspace(0,1, overlap, device=device)]
    mask_tensors += [torch.ones(ones_size, device=device)]
    return torch.cat(mask_tensors, 0).repeat(d2,1).rot90(rotate).repeat(c,1,1).unsqueeze(0)


# Apply blend masks to tiles
def mask_tile(tile, overlap, special=[0,0,0,0], side='left', s_sides=[False,False]):
    c, h, w = tile.size(1), tile.size(2), tile.size(3)
    base_mask = torch.ones_like(tile)
    if 'left' in side:
        base_mask = base_mask * create_lin_mask(overlap[3], special[3], s_sides[1], w, h, 0, c, tile.device)
    if 'bottom' in side:
        base_mask = base_mask * create_lin_mask(overlap[1], special[1], False, h, w, 1, c, tile.device)
    if 'right' in side:
        base_mask = base_mask * create_lin_mask(overlap[2], special[2], False, w, h, 2, c, tile.device)
    if 'top' in side:
        base_mask = base_mask * create_lin_mask(overlap[0], special[0], s_sides[0], h, w, 3, c, tile.device)
    # Apply mask to tile and return masked tile
    return tile * base_mask


# Put tensor tiles back together
def add_tiles(tiles, base_tensor, tile_coords, tile_size, overlap):
    # Collect required info for tiles that need different overlap values
    r, c = len(tile_coords[0]), len(tile_coords[1])
    f_ovlp = (tile_coords[0][r-1] - tile_coords[0][r-2], tile_coords[1][c-1] - tile_coords[1][c-2])

    column, row, t = 0, 0, 0
    for y in tile_coords[0]:
        for x in tile_coords[1]:
            mask_sides, c_overlap, s_sides = '', overlap.copy(), [False,False]
            if tile_coords[0] != [0]:
                if row == 0:
                    mask_sides += 'bottom'
                    c_overlap[1] = f_ovlp[0] if row == len(tile_coords[0]) - 2 else c_overlap[1]
                elif row > 0 and row < len(tile_coords[0]) -1:
                    mask_sides += 'bottom,top'
                    c_overlap[1] = f_ovlp[0] if row == len(tile_coords[0]) - 2 and f_ovlp[0] > 0 else c_overlap[1]
                elif row == len(tile_coords[0]) -1:
                    mask_sides += 'top'
                    c_overlap[0] = f_ovlp[0] if f_ovlp[0] > 0 else c_overlap[0]
                    s_sides[0] = True if f_ovlp[0] > 0 else False
            if tile_coords[1] != [0]:
                if column == 0:
                    mask_sides += ',right'
                    c_overlap[2] = f_ovlp[1] if column == len(tile_coords[1]) -2 else c_overlap[2]
                elif column > 0 and column < len(tile_coords[1]) -1:
                    mask_sides += ',right,left'
                    c_overlap[2] = f_ovlp[1] if column == len(tile_coords[1]) -2 and f_ovlp[1] > 0 else c_overlap[2]
                elif column == len(tile_coords[1]) -1:
                    mask_sides += ',left'
                    c_overlap[3] = f_ovlp[1] if f_ovlp[1] > 0 else c_overlap[3]
                    s_sides[1] = True if f_ovlp[1] > 0 else False

            tile = mask_tile(tiles[t], overlap.copy(), special=c_overlap, side=mask_sides, s_sides=s_sides)
            base_tensor[:, :, y:y+tile_size[0], x:x+tile_size[1]] = base_tensor[:, :, y:y+tile_size[0], x:x+tile_size[1]] + tile
            t+=1; column+=1
        row+=1; column=0
    return base_tensor


# Calculate tile coordinates
def get_tile_coords(d, tile_dim, overlap=0):
    c, tile_start, coords = 1, 0, [0]
    while tile_start + tile_dim < d:
        tile_start = int(tile_dim * (1-overlap)) * c
        coords.append(d - tile_dim) if tile_start + tile_dim >= d else coords.append(tile_start)
        c += 1
    return coords


# Calculates info required for tiling
def tile_setup(tile_size, overlap_percent, base_size):
    tile_size = [tile_size] * 2 if type(tile_size) is not tuple and type(tile_size) is not list else tile_size
    overlap_percent = [overlap_percent] * 2 if type(overlap_percent) is not tuple and type(overlap_percent) is not list else overlap_percent

    x_coords = get_tile_coords(base_size[1], tile_size[1], overlap_percent[1])
    y_coords = get_tile_coords(base_size[0], tile_size[0], overlap_percent[0])
    return (y_coords, x_coords), tile_size, [*([int(tile_size[0] * overlap_percent[0])] * 2), *([int(tile_size[1] * overlap_percent[1])] * 2)]


# Split tensor into tiles
def tile_tensor(tensor, tile_size, tile_overlap):
    tile_coords, tile_size, _ = tile_setup(tile_size, tile_overlap, (tensor.size(2), tensor.size(3)))
    tile_list = []
    [[tile_list.append(tensor[:, :, y:y + tile_size[0], x:x + tile_size[1]]) for x in tile_coords[1]] for y in tile_coords[0]]
    return tile_list


# Put tiles back into the original tensor
def rebuild_tensor(tiles, size, tile_size, tile_overlap):
    base_tensor = torch.zeros(size, device=tiles[0].device)
    tile_coords, tile_size, overlap = tile_setup(tile_size, tile_overlap, (base_tensor.size(2), base_tensor.size(3)))
    return add_tiles(tiles, base_tensor, tile_coords, tile_size, overlap)


# Calculate new tensor size if tile sizes are changed
def get_new_size(base_size, old_size, new_size):
    old_size = [old_size] * 2 if type(old_size) is not tuple and type(old_size) is not list else old_size
    new_size = [new_size] * 2 if type(new_size) is not tuple and type(new_size) is not list else new_size
    h, w = int((new_size[0] / old_size[0]) * base_size[2]), int((new_size[1] / old_size[1]) * base_size[3])
    return (base_size[0], base_size[1], h, w)


# Get tiling pattern & tile count
def get_tiling_info(tensor_size, tile_size, tile_overlap):
    tile_coords, tile_size, _ = tile_setup(tile_size, tile_overlap, (tensor_size[2], tensor_size[3]))
    return tile_size, (len(tile_coords[0]), len(tile_coords[1])), (len(tile_coords[0]) * len(tile_coords[1]))


# Handle tiling with spatial decorrelation
def handle_spectral(input, mod_list, scale_size, decay_power=1.0):
    scale_size = [scale_size] * 2 if type(scale_size) is not tuple and type(scale_size) is not list else scale_size
    if mod_list[0].h != scale_size[0] or mod_list[0].w != scale_size[1]:
        device = input.device if torch.is_tensor(input) else input[0].device
        mod_list[0].setup_scale(scale_size, decay_power, device)
    if torch.is_tensor(input):
        if input.dim() == 5:
           input = mod_list[0](input)
        else:
           input = mod_list[0].fft_image(input)
    else:
        if input[0].dim() == 5:
           input = [mod_list[0](tile) for tile in input]
        else:
           input = [mod_list[0].fft_image(tile) for tile in input]
    return input