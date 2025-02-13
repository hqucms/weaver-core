import torch
from gatr.interface import embed_vector


def get_num_spurions(
    beam_reference,
    add_time_reference,
    two_beams=True,
    add_xzplane=False,
    add_yzplane=False,
):
    """
    Compute how many reference multivectors/spurions a given configuration will have

    Parameters
    ----------
    beam_reference: str
        Different options for adding a beam_reference
        Options: "lightlike", "spacelike", "timelike", "xyplane"
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
    add_xzplane: bool
        Whether to add the x-z-plane as a reference to the network
    add_yzplane: bool
        Whether to add the y-z-plane as a reference to the network

    Returns
    -------
    num_spurions: int
        Number of spurions
    """
    num_spurions = 0
    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        num_spurions += 2 if two_beams else 1
    elif beam_reference == "xyplane":
        num_spurions += 1
    if add_xzplane:
        num_spurions += 1
    if add_yzplane:
        num_spurions += 1
    if add_time_reference:
        num_spurions += 1
    return num_spurions


def embed_spurions(
    beam_reference,
    add_time_reference,
    two_beams=True,
    add_xzplane=False,
    add_yzplane=False,
    device="cpu",
    dtype=torch.float32,
):
    """
    Construct a list of reference multivectors/spurions for symmetry breaking

    Parameters
    ----------
    beam_reference: str
        Different options for adding a beam_reference
        Options: "lightlike", "spacelike", "timelike", "xyplane"
    add_time_reference: bool
        Whether to add the time direction as a reference to the network
    two_beams: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
    add_xzplane: bool
        Whether to add the x-z-plane as a reference to the network
    add_yzplane: bool
        Whether to add the y-z-plane as a reference to the network
    device
    dtype

    Returns
    -------
    spurions: torch.tensor with shape (n_spurions, 16)
        spurion embedded as multivector object
    """
    kwargs = {"device": device, "dtype": dtype}

    if beam_reference in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_reference == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_reference == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_reference == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, **kwargs).reshape(1, 4)
        beam = embed_vector(beam)
        if two_beams:
            beam2 = beam.clone()
            beam2[..., 4] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)

    elif beam_reference == "xyplane":
        # add the x-y-plane, embedded as a bivector
        # convention for bivector components: [tx, ty, tz, xy, xz, yz]
        beam = torch.zeros(1, 16, **kwargs)
        beam[..., 8] = 1

    elif beam_reference is None:
        beam = torch.empty(0, 16, **kwargs)

    else:
        raise ValueError(f"beam_reference {beam_reference} not implemented")

    if add_xzplane:
        # add the x-z-plane, embedded as a bivector
        xzplane = torch.zeros(1, 16, **kwargs)
        xzplane[..., 10] = 1
    else:
        xzplane = torch.empty(0, 16, **kwargs)

    if add_yzplane:
        # add the y-z-plane, embedded as a bivector
        yzplane = torch.zeros(1, 16, **kwargs)
        yzplane[..., 9] = 1
    else:
        yzplane = torch.empty(0, 16, **kwargs)

    if add_time_reference:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, **kwargs).reshape(1, 4)
        time = embed_vector(time)
    else:
        time = torch.empty(0, 16, **kwargs)

    spurions = torch.cat((beam, xzplane, yzplane, time), dim=-2)
    return spurions
