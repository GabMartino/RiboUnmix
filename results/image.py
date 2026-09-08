import numpy as np
import matplotlib.pyplot as plt

# Match the LaTeX-style figures used elsewhere in the repository.  Latin
# Modern Roman is preferred when installed, with Computer Modern and
# DejaVu Serif as fallbacks; mathtext uses matching Computer Modern glyphs.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # Keep text editable in the SVG.
        "svg.fonttype": "none",
    }
)

rng = np.random.default_rng(42)
n_codons = 220
position = np.arange(1, n_codons + 1)


def gaussian_peak(center, height, width):
    return height * np.exp(
        -0.5 * ((position - center) / width) ** 2
    )


def sharp_spikes(events, noise_scale=0.0):
    """Build a noisy real-like profile from narrow, non-Gaussian spikes."""
    profile = np.ones(n_codons, dtype=float)
    if noise_scale:
        profile += rng.normal(0.0, noise_scale, size=n_codons)
    for center, amplitude, left_width, right_width in events:
        distance = position - center
        width = np.where(distance < 0, left_width, right_width)
        contribution = amplitude * np.maximum(1.0 - np.abs(distance) / width, 0.0)
        profile += contribution
    return np.clip(profile, 0.05, None)


# Shared sequence-dependent biological profile
L_bio = (
    1.2
    + 0.35 * np.sin(position / 17)
    + 0.20 * np.cos(position / 8)
)

# Local biological pauses
for center, height, width in [
    (28, 7, 1.4),
    (54, 13, 1.0),
    (91, 8, 1.8),
    (126, 18, 0.9),
    (158, 10, 1.5),
    (194, 14, 1.1),
]:
    L_bio += gaussian_peak(center, height, width)

L_bio = np.clip(L_bio, 0.05, None)


# Different protocol-dependent distortions.  The narrow peaks and dips mimic
# positional gamma structure in real datasets; centering changes only the
# overall gauge, not the spiky shape.
gamma_A_raw = sharp_spikes(
    [
        (19, 1.5, 1.0, 2.0),
        (34, 0.65, 1.0, 1.0),
        (51, 0.95, 2.0, 1.0),
        (72, 1.8, 2.0, 1.0),
        (81, -0.35, 1.0, 1.0),
        (96, 0.7, 1.0, 2.0),
        (113, 0.9, 1.0, 2.0),
        (121, -0.4, 1.0, 1.0),
        (145, 0.75, 2.0, 1.0),
        (158, 1.3, 2.0, 1.0),
        (176, -0.3, 1.0, 2.0),
        (188, 0.8, 1.0, 1.0),
        (207, 1.1, 1.0, 1.0),
        (45, -0.55, 1.0, 2.0),
        (137, -0.45, 2.0, 1.0),
        (216, -0.3, 1.0, 1.0),
    ],
    noise_scale=0.055,
)

gamma_B_raw = sharp_spikes(
    [
        (31, 1.1, 1.0, 1.0),
        (42, 1.6, 1.0, 2.0),
        (57, -0.35, 1.0, 1.0),
        (74, 0.75, 2.0, 1.0),
        (88, 0.8, 2.0, 1.0),
        (103, 0.6, 1.0, 1.0),
        (126, 1.8, 1.0, 1.0),
        (137, -0.4, 1.0, 2.0),
        (148, 0.85, 1.0, 1.0),
        (171, 1.0, 1.0, 2.0),
        (181, -0.3, 2.0, 1.0),
        (192, 0.7, 1.0, 2.0),
        (203, 1.4, 2.0, 1.0),
        (67, -0.5, 1.0, 1.0),
        (154, -0.6, 2.0, 1.0),
        (213, -0.25, 1.0, 1.0),
    ],
    noise_scale=0.06,
)

gamma_C = (
    1.0
    + gaussian_peak(63, 1.4, 1.8)
    + gaussian_peak(146, 1.1, 2.2)
    + gaussian_peak(184, 1.7, 1.5)
)


def geometric_mean_one(gamma):
    """Apply the real-case gamma gauge: mean(log(gamma)) == 0."""
    return gamma / np.exp(np.mean(np.log(gamma)))


gamma_A = geometric_mean_one(gamma_A_raw)
gamma_B = geometric_mean_one(gamma_B_raw)
gamma_C_centered = geometric_mean_one(gamma_C)

# These are the real-case constraints: positive gamma and unit geometric
# mean (equivalently, zero mean in log-gamma space).
assert np.all(gamma_A > 0) and np.all(gamma_B > 0)
assert np.isclose(np.mean(np.log(gamma_A)), 0.0)
assert np.isclose(np.mean(np.log(gamma_B)), 0.0)


def sample_nb2(mu, alpha=0.35):
    """NB2 distribution: variance = mu + alpha * mu²."""
    size = 1.0 / alpha
    probability = size / (size + mu)
    return rng.negative_binomial(size, probability)


counts_A = sample_nb2(2.2 * L_bio * gamma_A)
counts_B = sample_nb2(2.8 * L_bio * gamma_B)
counts_C = sample_nb2(2.5 * L_bio * gamma_C_centered)


def save_riboseq_profile(counts, color, title, filename):
    fig, ax = plt.subplots(figsize=(3.4, 1.25))

    # An unsmoothed step line keeps codon counts discrete
    ax.step(
        position,
        counts,
        where="mid",
        color=color,
        linewidth=0.9,
    )

    ax.fill_between(
        position,
        counts,
        step="mid",
        color=color,
        alpha=0.12,
        linewidth=0,
    )

    ax.set_xlim(1, n_codons)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Codon position", fontsize=7)
    ax.set_ylabel("RPF counts", fontsize=7)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold")

    ax.set_xticks([1, 50, 100, 150, 200])
    ax.tick_params(axis="both", labelsize=6, length=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")

    ax.grid(axis="y", color="#E2E8F0", linewidth=0.45)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.3)
    fig.savefig(filename, transparent=True, bbox_inches="tight")
    plt.close(fig)


save_riboseq_profile(
    counts_A,
    color="#D97706",
    title="",
    filename="riboseq_dataset_A.svg",
)

save_riboseq_profile(
    counts_B,
    color="#6D5BD0",
    title="",
    filename="riboseq_dataset_B.svg",
)

save_riboseq_profile(
    counts_C,
    color="#0F766E",
    title="",
    filename="riboseq_dataset_C.svg",
)


def save_gamma_correction(gamma, color, filename):
    fig, ax = plt.subplots(figsize=(3.4, 1.25))
    ax.plot(position, gamma, color=color, linewidth=0.9)
    ax.axhline(1.0, color="#94A3B8", linewidth=0.45, linestyle="--")
    ax.set_xlim(1, n_codons)
    ax.set_ylim(0, 2.45)
    ax.set_xlabel("Codon position", fontsize=7)
    # Use the Unicode glyph here: this avoids a cmmi10 SVG export quirk that
    # can make mathtext gamma appear as a degree-like symbol.
    ax.set_ylabel("γ", fontsize=7, labelpad=1)
    ax.tick_params(axis="both", labelsize=6, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([1, 50, 100, 150, 200])
    ax.set_yticks([0, 1, 2])
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.45, linestyle="--")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.45)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(filename, transparent=True, bbox_inches="tight")
    plt.close(fig)


save_gamma_correction(gamma_A, "#D97706", "gamma_dataset_A.svg")
save_gamma_correction(gamma_B, "#6D5BD0", "gamma_dataset_B.svg")
