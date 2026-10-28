"""How much of each dataset a training run reads: the data mixture.

Without a mixture every block of every recording is read once per epoch, so the mix follows
the data: in index 2026-10-27b one dataset (i-care, ICU EEG) is 61 % of the tokens.

UniMax (Chung et al. 2023, "UniMax: Fairer and more effective language sampling for
large-scale multilingual pretraining", Algorithm 1) gives each source an equal share of a
token budget, but at most `max_epochs` passes over it; what a small source cannot use goes to
the larger ones (the equal share is computed again over the sources left). Here it works on
two levels: clusters of datasets (groups the experiment names, e.g. "12-lead ECG", "sleep
PSG"), then the datasets in each cluster with the cluster's share. Repeating data up to about
4 epochs costs little (Muennighoff et al. 2023, "Scaling data-constrained language models").

The result is a number of passes per dataset: 2.6 means every block of the dataset twice and
a random 60 % of its blocks a third time in one epoch; 0.3 means a random 30 % of its blocks.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from biosignal_data.training.loader import Selection


def unimax(sizes: Mapping[str, float], budget: float, max_epochs: float) -> dict[str, float]:
    """Tokens for each source: an equal share of `budget`, but at most `max_epochs` times
    its size; the smallest sources first, so what they cannot use goes to the rest."""
    if budget <= 0 or max_epochs <= 0:
        raise ValueError("the budget and max_epochs must be positive")
    out: dict[str, float] = {}
    left = float(budget)
    ordered = sorted(sizes.items(), key=lambda item: (item[1], item[0]))
    for i, (name, size) in enumerate(ordered):
        out[name] = min(left / (len(ordered) - i), max_epochs * size)
        left -= out[name]
    return out


def passes(
    sizes: Mapping[str, float],
    clusters: Mapping[str, Iterable[str]],
    budget: float,
    max_epochs: float,
) -> dict[str, float]:
    """Passes over each dataset of `sizes` (tokens) in one epoch of `budget` tokens:
    UniMax over the clusters, then over the datasets of each cluster. Every dataset of
    `sizes` must be in exactly one cluster; clusters may name datasets that `sizes` has not
    (an experiment can select fewer)."""
    of: dict[str, str] = {}
    for cluster, members in clusters.items():
        for dataset in members:
            if dataset in of:
                raise ValueError(f"{dataset} is in clusters {of[dataset]} and {cluster}")
            of[dataset] = cluster
    missing = sorted(set(sizes) - set(of))
    if missing:
        raise ValueError(f"datasets in no cluster: {', '.join(missing)}")
    members_of: dict[str, dict[str, float]] = {}
    for dataset, size in sizes.items():
        if size > 0:
            members_of.setdefault(of[dataset], {})[dataset] = size
    shares = unimax({c: sum(m.values()) for c, m in members_of.items()}, budget, max_epochs)
    out = dict.fromkeys(sizes, 0.0)
    for cluster, members in members_of.items():
        for dataset, tokens in unimax(members, shares[cluster], max_epochs).items():
            out[dataset] = tokens / members[dataset]
    return out


def dataset_tokens(selection: Selection, samples_per_token: int) -> dict[str, float]:
    """The tokens of each dataset of a selection: the samples of its selected channels
    (duration x sampling rate) over `samples_per_token`."""
    out: dict[str, float] = {}
    for r in selection.recordings:
        samples = sum(c.sampling_rate for c in r.channels) * r.duration_s
        out[r.dataset_id] = out.get(r.dataset_id, 0.0) + samples / samples_per_token
    return out
