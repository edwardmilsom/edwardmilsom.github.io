import numpy as np
import matplotlib.pyplot as plt

# Explicit double precision grids
sigma = np.linspace(0.0, 1.0, 4001, dtype=np.float64)
sigma_log = np.geomspace(np.float64(1e-4), np.float64(1), 4000, dtype=np.float64)

lams = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1e-0]
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

# Quintic Newton-Schulz polynomial from the blog post:
# P(X) = (15/8)X - (5/4)(XX^T)X + (3/8)(XX^T)^2X
QUINTIC_NS_COEFFS = [
    (np.float64(15.0 / 8.0), np.float64(-5.0 / 4.0), np.float64(3.0 / 8.0))
]

def get_coeffs(method="polar_express", n_steps=5):
    if method == "polar_express":
        if n_steps != 5:
            raise ValueError("polar_express is only defined here for n_steps=5.")
        return POLAR_EXPRESS_COEFFS
    elif method == "cubic_ns":
        return CUBIC_NS_COEFFS[:n_steps]
    elif method == "quintic_ns":
        return QUINTIC_NS_COEFFS * n_steps
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

def _as_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]

def _resolve_runs(method, n_steps):
    methods = _as_sequence(method)
    steps = _as_sequence(n_steps)

    if len(methods) == 1 and len(steps) > 1:
        methods = methods * len(steps)
    elif len(steps) == 1 and len(methods) > 1:
        steps = steps * len(methods)
    elif len(methods) != len(steps):
        raise ValueError(
            "method and n_steps must be scalars, or lists of the same length, "
            "or one of them must be a scalar that can be broadcast."
        )

    return list(zip(methods, steps))

def make_plots(method="cubic_ns", n_steps=5, prefix="polar_maps"):
    runs = _resolve_runs(method, n_steps)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lam_to_color = {lam: color_cycle[i % len(color_cycle)] for i, lam in enumerate(lams)}
    lam_to_color[np.float64(0.0)] = "black"

    exact_polar = np.ones_like(sigma, dtype=np.float64)
    exact_polar[0] = np.float64(0.0)

    def run_label(run_method, run_steps):
        if run_method == "polar_express":
            return "Polar Express"
        if run_method == "cubic_ns":
            return f"Cubic Newton-Schulz ({run_steps} iters)"
        if run_method == "quintic_ns":
            return fr"Quintic NS-{run_steps} Muon ($\lambda=0$)"
        return f"{run_method} ({run_steps})"

    ns_runs = []
    for run_method, run_steps in runs:
        coeffs = get_coeffs(method=run_method, n_steps=run_steps)
        ns_runs.append(
            {
                "method": run_method,
                "n_steps": run_steps,
                "label": run_label(run_method, run_steps),
                "coeffs": coeffs,
                "linear": safe_vals(ns_map(sigma, coeffs)),
                "log": safe_vals(ns_map(sigma_log, coeffs)),
            }
        )
    ns_shades = [str(x) for x in np.linspace(0.75, 0.15, max(len(ns_runs), 1), dtype=np.float64)]

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
        # for lam in lams64:
        #     y = exact_tikhonov(x, lam) if axis_mode != "linear" else exact_tikhonov(sigma, lam)
        #     if y_floor is not None:
        #         y = np.maximum(y, y_floor)
        #     plot_fn(
        #         x if axis_mode != "linear" else sigma,
        #         y,
        #         linestyle="--",
        #         color=lam_to_color[float(lam)],
        #         label=fr"$\lambda={lam:g}$",
        #     )

        # unregularized NS
        for idx, run in enumerate(ns_runs):
            y_ns = run["linear"] if axis_mode == "linear" else run["log"]
            if y_floor is not None:
                y_ns = np.maximum(y_ns, y_floor)
            if run["method"] == "quintic_ns" and run["n_steps"] == 5:
                ns_color = "black"
            else:
                ns_color = ns_shades[idx % len(ns_shades)]
            plot_fn(
                x if axis_mode != "linear" else sigma,
                y_ns,
                linewidth=2.5,
                label=run["label"],
                color=ns_color,
            )

        # shifted NS Tikhonov: dotted
        for run in ns_runs:
            for lam in lams64:
                if lam == 0.0:
                    continue
                y = (
                    gram_ns_tikhonov(x, lam, run["coeffs"])
                    if axis_mode != "linear"
                    else gram_ns_tikhonov(sigma, lam, run["coeffs"])
                )
                y = safe_vals(y)
                if y_floor is not None:
                    y = np.where(np.isfinite(y), np.maximum(y, y_floor), np.nan)
                lam_label = fr"MuonTikh ($\lambda = {lam:g}$)"
                plot_fn(
                    x if axis_mode != "linear" else sigma,
                    y,
                    linestyle=":",
                    color=lam_to_color[float(lam)],
                    label=lam_label,
                )

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
    logx_path = f"./muontikh_polar_logx.svg"
    loglog_path = f"./muontikh_polar_loglog.svg"

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
        xlim=(1e-4, 1),
        ylim=(0, 1.1),
        title="Applying Tikhonov Regularisation to NS Muon (MuonTikh)",
        outpath=logx_path,
    )

    draw_plot(
        sigma_log,
        axis_mode="loglog",
        xlim=(1e-4, 1),
        ylim=(1e-4, 2),
        title="Applying Tikhonov Regularisation to NS Muon (MuonTikh)",
        outpath=loglog_path,
        y_floor=np.float64(1e-6),
    )

if __name__ == "__main__":
    # Options:
    #   method="polar_express"   uses the 5 blogpost coefficients
    #   method="cubic_ns"        uses x <- 1.5 x - 0.5 x^3 repeated n_steps times
    #   method="quintic_ns"      uses the quintic polynomial from the blog post repeated n_steps times
    #   method=["cubic_ns", "quintic_ns"], n_steps=[5, 1]
    #                            overlays multiple methods on the same plots
    make_plots(method="quintic_ns", n_steps=5, prefix="polar_maps")
