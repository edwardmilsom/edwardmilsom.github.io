import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import comb

# Grid of input singular values: all powers of 10 from 1e-1 to 1e-8
sigmas = [np.float64(10.0**(-j)) for j in range(1, 9,2)]
k_max = 16

def taylor_coeffs_inv_sqrt(m):
    return [np.float64(comb(2*j, j) / (4**j)) for j in range(m + 1)]

def poly_coeffs_from_order(max_power):
    m = (max_power - 1) // 2
    c = taylor_coeffs_inv_sqrt(m)
    q = np.zeros(m + 1, dtype=np.float64)
    for j in range(m + 1):
        for ell in range(j + 1):
            q[ell] += c[j] * comb(j, ell) * ((-1) ** ell)
    return q  # coefficients for x, x^3, ..., x^(2m+1)

def apply_odd_poly(x, odd_coeffs):
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x, dtype=np.float64)
    xpow = x.copy()
    for a in odd_coeffs:
        y += a * xpow
        xpow = xpow * x * x
    return y

def iterates_from_poly(sig, odd_coeffs, k_max):
    x = np.float64(sig)
    ys = [x]
    for _ in range(k_max):
        x = apply_odd_poly(np.array([x], dtype=np.float64), odd_coeffs)[0]
        ys.append(x)
    return np.array(ys, dtype=np.float64)

def lambda_from_exact_tikh(sig, y):
    sig = np.asarray(sig, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return sig**2 * (1.0 / y**2 - 1.0)

orders = {
    "Cubic (3rd)": 3,
    "Quintic (5th)": 5,
    "Septic (7th)": 7,
    "Nonic (9th)": 9,
    "11th-order": 11,
}
order_coeffs = {label: poly_coeffs_from_order(order) for label, order in orders.items()}
ks = np.arange(k_max + 1, dtype=int)

fig, axes = plt.subplots(2, 2, figsize=(9, 10), sharex=True, sharey=True)
axes = axes.ravel()

records = []

for ax, sigma in zip(axes, sigmas):
    for label, coeffs in order_coeffs.items():
        ys = iterates_from_poly(sigma, coeffs, k_max)
        lams = lambda_from_exact_tikh(sigma, ys)
        lams_plot = np.where((np.isfinite(lams)) & (lams > 0), lams, np.nan)
        data_line = ax.semilogy(
            ks,
            lams_plot,
            marker="o",
            markersize=3,
            # linestyle="None",
            label=f"{label} Actual",
        )[0]

        alpha = float(coeffs[0])
        if alpha != 0.0:
            lam_theory = np.power(alpha, -2 * ks, dtype=np.float64)
        else:
            lam_theory = np.full_like(ks, np.nan, dtype=np.float64)
        ax.semilogy(
            ks,
            lam_theory,
            color=data_line.get_color(),
            linestyle="--",
            linewidth=1.2,
            label=f"{label} Theory",
        )

        for k, yk, lam in zip(ks, ys, lams):
            records.append({
                "sigma": float(sigma),
                "k": int(k),
                "method": label,
                "ns_output_yk": float(yk),
                "equivalent_lambda": float(lam),
            })

    exp = int(np.log10(float(sigma)))
    ax.set_title(rf"Input singular value $\sigma = 10^{{{exp}}}$", fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlabel(r"Newton-Schulz iteration count $k$")

for ax in axes[:2]:
    ax.tick_params(axis="x", labelbottom=True)

for ax in axes[::2]:
    ax.set_ylabel(r"$\lambda_k$ where $Q_\lambda(\sigma) = Q_{\text{NS-}k}(\sigma)$")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels)//2,
    frameon=True,
    bbox_to_anchor=(0.5, 0.93),
)

fig.suptitle(
    r"Newton-Schulz iterations and Tikhonov regularisation are linearly related"
    "\n"
    r"in the limit of small input singular values",
    y=0.985,
    fontsize=15,
)

fig.tight_layout(rect=[0, 0, 1, 0.94])

fig.savefig("tikh_ns_iters_predicted_law.svg", bbox_inches="tight")
# plt.show()

# pd.DataFrame(records).to_csv("ns_equivalent_lambda_grid_sigmas.csv", index=False)
