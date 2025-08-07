import numpy as np
import matplotlib.pyplot as plt

# --- Exponential-decay-to-floor scaling and bifurcation check ---

def compute_theta(f_star: float,
                  F_max: float,
                  s_min: float) -> float:
    """
    Compute the decay constant θ so that the fixed-point condition
        f_star = F_max * [s_min + (1 - s_min) * exp(-f_star/θ)]
    is satisfied exactly.
    """
    s_star = f_star / F_max
    if not (s_min < s_star < 1.0):
        raise ValueError("Require s_min < f_star/F_max < 1.0")
    ratio = (s_star - s_min) / (1.0 - s_min)
    return -f_star / np.log(ratio)


def exp_scale_and_derivative(f: float, theta: float, s_min: float) -> tuple:
    """
    Returns the scale s(f) and its derivative s'(f) for exponential-decay-to-floor.
    s(f) = s_min + (1 - s_min)*exp(-f/theta)
    s'(f) = -(1 - s_min)/theta * exp(-f/theta)
    """
    exp_term = np.exp(-f / theta)
    s = s_min + (1 - s_min) * exp_term
    ds_df = -(1 - s_min) / theta * exp_term
    return s, ds_df


if __name__ == "__main__":
    # Example parameters
    F_max   = 5.0   # N
    s_min   = 0.06    # floor
    f_star  = 0.4    # desired equilibrium

    # Compute θ
    theta = compute_theta(f_star, F_max, s_min)
    print(f"Computed θ = {theta:.4f}")

    # Evaluate scale and derivative at f_star
    s_star, ds_df_star = exp_scale_and_derivative(f_star, theta, s_min)
    g_prime = F_max * ds_df_star
    print(f"At f* = {f_star} N:")
    print(f"  s(f*) = {s_star:.4f}")
    print(f"  s'(f*) = {ds_df_star:.4f}")
    print(f"  g'(f*) = F_max * s'(f*) = {g_prime:.4f}")

    # Bifurcation check
    if abs(g_prime) < 1.0:
        print("--> Stable fixed point (|g'(f*)| < 1)")
    else:
        print("--> Unstable: bifurcation/oscillation likely (|g'(f*)| >= 1)")

    # Optional: simulate a few iterations to illustrate
    def g(f):
        s, _ = exp_scale_and_derivative(f, theta, s_min)
        return F_max * s

    n_steps = 20
    f_vals = np.zeros(n_steps)
    f_vals[0] = F_max
    for i in range(1, n_steps):
        f_vals[i] = g(f_vals[i-1])

    # Plot iteration
    plt.figure(figsize=(6, 3))
    plt.plot(f_vals, 'o-', label='f_n')
    plt.hlines(f_star, 0, n_steps-1, colors='gray', linestyles='--', label='f*')
    plt.title('Iteration Sequence')
    plt.xlabel('Iteration n')
    plt.ylabel('f_n [N]')
    plt.legend()
    plt.tight_layout()
    plt.show()