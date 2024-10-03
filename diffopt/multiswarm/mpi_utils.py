import math

import numpy as np

try:
    from mpi4py.MPI import COMM_WORLD
except ImportError:
    COMM_WORLD = None


def split_subcomms(num_groups=None, ranks_per_group=None,
                   comm=COMM_WORLD):
    """
    Split comm into sub-comms (not grouped by nodes)

    Parameters
    ----------
    num_groups : int, optional
        Specify the number of evenly divided groups of subcomms
    ranks_per_group : list[int], optional
        Specify the number of ranks given to each sub-comm
    comm : MPI.Comm, optional
        Specify a sub-communicator to split into sub-sub-communicators

    Returns
    -------
    subcomm: MPI.Comm
        The sub-comm that now controls this process
    num_groups: int
        The number of groups of subcomms (same as input if not None)
    group_rank: int
        The rank of this group (0 <= subcomm_rank < num_subcomms)
    """
    assert comm is not None, "Cannot split subcomms without mpi4py"
    main_msg = "Specify either num_subcomms OR ranks_per_subcomm"
    sumrps_msg = "The sum of ranks_per_subcomm must equal comm.size"
    nsub_msg = "Cannot create more subcomms than there are ranks"
    if num_groups is not None:
        assert ranks_per_group is None, main_msg
        assert (comm.size >= num_groups), nsub_msg
        num_groups = int(num_groups)
        subnames = (np.ones(math.ceil(comm.size / num_groups))[None, :]
                    * np.arange(num_groups)[:, None])[:comm.size]
        subnames = subnames.ravel().astype(int)
    else:
        assert ranks_per_group is not None, main_msg
        assert sum(ranks_per_group) == comm.size, sumrps_msg
        num_groups = len(ranks_per_group)
        subnames = np.repeat(np.arange(num_groups), ranks_per_group)

    subname = str(np.array_split(subnames, comm.size)[comm.rank][0])

    nodelist = comm.allgather(subname)
    unique_nodelist = sorted(list(set(nodelist)))
    node_number = unique_nodelist.index(subname)
    intra_node_id = len([i for i in nodelist[:comm.rank] if i == subname])
    comm.Barrier()

    rankinfo = (comm.rank, intra_node_id, node_number)
    infolist = comm.allgather(rankinfo)
    sorted_infolist = sorted(infolist, key=lambda x: x[1])
    sorted_infolist = sorted(sorted_infolist, key=lambda x: x[2])

    sub_comm = comm.Split(color=node_number)
    sub_comm.Set_name(f"{comm.name}.{subname}".replace(
        "MPI_COMM_WORLD.", ""))

    # sub_comm.Free()  # Sometimes this cleanup helps prevent memory leaks
    return sub_comm, num_groups, int(subname)
