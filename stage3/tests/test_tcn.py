import torch

from stage3.model.tcn import TemporalConvNet


def test_whole_and_replicate_padded_valid_region_agree():
    torch.manual_seed(1)
    model = TemporalConvNet(dim=16, dilations=(1, 2, 4), dropout=0).eval()
    short = torch.randn(1, 21, 16)
    padded = torch.cat((short, short[:, -1:].expand(-1, 9, -1)), dim=1)
    with torch.no_grad():
        expected = model(short)
        actual = model(padded)[:, :21]
    # Future context differs only inside the right receptive-field margin.
    assert torch.allclose(actual[:, :-14], expected[:, :-14], atol=1e-6)
    assert model.receptive_field == 15
