import torch


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
    # padd
    batch = [torch.Tensor(t).to(device) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    # compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask
