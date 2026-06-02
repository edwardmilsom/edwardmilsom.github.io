import numpy as np
import matplotlib.pyplot as plt

# Explicit double precision grids
sigma = np.linspace(0.0, 1.0, 4001, dtype=np.float64)
sigma_log = np.geomspace(np.float64(1e-6), np.float64(100), 4000, dtype=np.float64)

lams = [1e-6,  1e-4,  1e-2, 1e-1, 1e-0, 1e1]
lams64 = [np.float64(x) for x in lams]

POLAR_EXPRESS_COEFFS = [
    (8.123737, -22.232240, 16.373715),
    (4.026529, -2.776323, 0.514551),
    (3.870284, -2.739120, 0.520999),
    (3.253351, -2.343223, 0.481420),
    (2.300652, -1.668904, 0.418807),
]
POLAR_EXPRESS_COEFFS = [(np.float64(a), np.float64(b), np.float64(c)) for a, b, c in POLAR_EXPRESS_COEFFS]

# Classical cubic Newton-Schulz:
# x_{k+1} = (3/2)x_k - (1/2)x_k^3
CUBIC_STEPS = 5
CUBIC_NS_COEFFS = [(np.float64(1.5), np.float64(-0.5), np.float64(0.0))] * CUBIC_STEPS

def get_coeffs(method="polar_express", n_steps=5):
    if method == "polar_express":
        if n_steps != 5:
            raise ValueError("polar_express is only defined here for n_steps=5.")
        return POLAR_EXPRESS_COEFFS
    elif method == "cubic_ns":
        return CUBIC_NS_COEFFS[:n_steps]
    else:
        raise ValueError(f"Unknown method: {method}")

def p(x, a, b, c):
    x = np.asarray(x, dtype=np.float64)
    return a*x + b*x**3 + c*x**5

def h(r, a, b, c):
    r = np.asarray(r, dtype=np.float64)
    return a + b*r + c*r**2

def ns_map(sig, coeffs):
    x = np.asarray(sig, dtype=np.float64).copy()
    for a, b, c in coeffs:
        x = p(x, a, b, c)
    return x

def exact_tikhonov(sig, lam):
    sig = np.asarray(sig, dtype=np.float64)
    lam = np.float64(lam)
    return sig / np.sqrt(sig**2 + lam)

def gram_ns_tikhonov(sig, lam, coeffs):
    sig = np.asarray(sig, dtype=np.float64)
    lam = np.float64(lam)
    r = sig**2 + lam
    q = np.ones_like(sig, dtype=np.float64)
    for a, b, c in coeffs:
        z = h(r, a, b, c)
        q = q * z
        r = r * z**2
    return sig * q

def safe_vals(y):
    y = np.asarray(y, dtype=np.float64)
    return np.where(np.isfinite(y), y, np.nan)

def make_plots(method="cubic_ns", n_steps=5, prefix="polar_maps"):
    coeffs = get_coeffs(method=method, n_steps=n_steps)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lam_to_color = {lam: color_cycle[i % len(color_cycle)] for i, lam in enumerate(lams)}

    exact_polar = np.ones_like(sigma, dtype=np.float64)
    exact_polar[0] = np.float64(0.0)

    ns_unreg = safe_vals(ns_map(sigma, coeffs))
    ns_unreg_log = safe_vals(ns_map(sigma_log, coeffs))

    method_label = "Polar Express" if method == "polar_express" else f"Cubic Newton-Schulz ({n_steps} iters)"

    def draw_plot(x, axis_mode="linear", xlim=None, ylim=None, title="", outpath="plot.svg", y_floor=None):
        plt.figure(figsize=(6, 5))

        if axis_mode == "linear":
            plot_fn = plt.plot
        elif axis_mode == "semilogx":
            plot_fn = plt.semilogx
        elif axis_mode == "loglog":
            plot_fn = plt.loglog
        else:
            raise ValueError(f"Unknown axis_mode: {axis_mode}")

        # exact polar
        if axis_mode == "linear":
            plot_fn(sigma, exact_polar, linewidth=2.5, label=fr"$\lambda=0$ (Exact polar)")
        else:
            # avoid y=0 on log axis
            plot_fn(x, np.ones_like(x, dtype=np.float64), linewidth=2.5, label=fr"$\lambda=0$ (Exact polar)")

        # exact Tikhonov: dashed
        for lam in lams64:
            y = exact_tikhonov(x, lam) if axis_mode != "linear" else exact_tikhonov(sigma, lam)
            if y_floor is not None:
                y = np.maximum(y, y_floor)
            plot_fn(
                x if axis_mode != "linear" else sigma,
                y,
                linestyle="--",
                color=lam_to_color[float(lam)],
                label=fr"$\lambda={lam:g}$",
            )

        # unregularized NS
        # y_ns = ns_unreg if axis_mode == "linear" else ns_unreg_log
        # if y_floor is not None:
        #     y_ns = np.maximum(y_ns, y_floor)
        # plot_fn(
        #     x if axis_mode != "linear" else sigma,
        #     y_ns,
        #     linewidth=2.5,
        #     label=method_label,
        #     color="black",
        # )

        # shifted NS Tikhonov: dotted
        # for lam in lams64:
        #     y = gram_ns_tikhonov(x, lam, coeffs) if axis_mode != "linear" else gram_ns_tikhonov(sigma, lam, coeffs)
        #     y = safe_vals(y)
        #     if y_floor is not None:
        #         y = np.where(np.isfinite(y), np.maximum(y, y_floor), np.nan)
        #     plot_fn(
        #         x if axis_mode != "linear" else sigma,
        #         y,
        #         linestyle=":",
        #         color=lam_to_color[float(lam)],
        #         label=fr"NS Tik. $\lambda={lam:g}$",
        #     )

        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)

        xlabel = "Input singular value"
        if axis_mode in ("semilogx", "loglog"):
            xlabel += " (log scale)"
        ylabel = "Output singular value"
        if axis_mode == "loglog":
            ylabel += " (log scale)"
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3, which="both" if axis_mode != "linear" else "major")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches="tight")

    # full_path = f"{prefix}_{method}_full.svg"
    # zoom_path = f"{prefix}_{method}_zoom.svg"
    logx_path = f"./tikh_polar_logx.svg"
    loglog_path = f"./tikh_polar_loglog.svg"

    # draw_plot(
    #     sigma,
    #     axis_mode="linear",
    #     xlim=(0, 1),
    #     ylim=(0, 1.25),
    #     title=f"Scalar maps on singular values in [0,1] - {method_label}",
    #     outpath=full_path,
    # )

    # draw_plot(
    #     sigma,
    #     axis_mode="linear",
    #     xlim=(0, 0.08),
    #     ylim=(0, 1.1),
    #     title=f"Zoom near zero - {method_label}",
    #     outpath=zoom_path,
    # )

    draw_plot(
        sigma_log,
        axis_mode="semilogx",
        xlim=(1e-6, 100),
        ylim=(0, 1.1),
        title=f"Tikhonov Regularised Polar Maps (Log x-axis)",
        outpath=logx_path,
    )

    draw_plot(
        sigma_log,
        axis_mode="loglog",
        xlim=(1e-6, 100),
        ylim=(1e-6, 2),
        title=f"Tikhonov Regularised Polar Maps (Log-Log)",
        outpath=loglog_path,
        y_floor=np.float64(1e-6),
    )

if __name__ == "__main__":
    # Options:
    #   method="polar_express"   uses the 5 blogpost coefficients
    #   method="cubic_ns"        uses x <- 1.5 x - 0.5 x^3 repeated n_steps times
    make_plots(method="cubic_ns", n_steps=CUBIC_STEPS, prefix="polar_maps")
