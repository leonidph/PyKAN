#include "KANLinear.h"

KANLinearImpl::KANLinearImpl(
    int in_features_,
    int out_features_,
    int grid_size_,
    int spline_order_,
    float scale_noise_,
    float scale_base_,
    float scale_spline_,
    bool enable_standalone_scale_spline_,
    float grid_eps_,
    std::pair<float,float> grid_range
)
: in_features(in_features_),
  out_features(out_features_),
  grid_size(grid_size_),
  spline_order(spline_order_),
  scale_noise(scale_noise_),
  scale_base(scale_base_),
  scale_spline(scale_spline_),
  enable_standalone_scale_spline(enable_standalone_scale_spline_),
  grid_eps(grid_eps_),
  base_activation(torch::nn::SiLU())
{
    float h = (grid_range.second - grid_range.first) / grid_size;

    auto idx = torch::arange(-spline_order, grid_size + spline_order + 1, torch::kFloat);
    auto g = idx * h + grid_range.first;
    g = g.expand({in_features, g.size(0)}).contiguous();

    grid = register_buffer("grid", g);

    base_weight = register_parameter(
        "base_weight",
        torch::randn({out_features, in_features})
    );

    spline_weight = register_parameter(
        "spline_weight",
        torch::randn({out_features, in_features, grid_size + spline_order})
    );

    if (enable_standalone_scale_spline) {
        spline_scaler = register_parameter(
            "spline_scaler",
            torch::randn({out_features, in_features})
        );
    }

    reset_parameters();
}

void KANLinearImpl::reset_parameters() {
    torch::nn::init::kaiming_uniform_(base_weight, std::sqrt(5.0f) * scale_base);

    auto noise = (torch::rand({grid_size + 1, in_features, out_features}) - 0.5f)
                 * scale_noise / grid_size;

    auto coeff = curve2coeff(
        grid.transpose(0,1).slice(0, spline_order, -spline_order),
        noise
    );

    if (!enable_standalone_scale_spline)
        coeff = coeff * scale_spline;

    spline_weight.data().copy_(coeff);

    if (enable_standalone_scale_spline) {
        torch::nn::init::kaiming_uniform_(spline_scaler, std::sqrt(5.0f) * scale_spline);
    }
}

torch::Tensor KANLinearImpl::b_splines(const torch::Tensor& x) {
    auto x_exp = x.unsqueeze(-1); // (B, in_features, 1)
    auto g = grid;                // (in_features, grid_size + 2*spline_order + 1)

    auto bases = ((x_exp >= g.slice(1, 0, -1)) &
                  (x_exp <  g.slice(1, 1))).to(x.dtype());

    for (int k = 1; k <= spline_order; k++) {
        auto left  = (x_exp - g.slice(1, 0, -(k+1))) /
                     (g.slice(1, k, -1) - g.slice(1, 0, -(k+1)) + grid_eps);

        auto right = (g.slice(1, k+1, -1) - x_exp) /
                     (g.slice(1, k+1, -1) - g.slice(1, 1, -(k+1)) + grid_eps);

        bases = left * bases.slice(2, 0, -1) + right * bases.slice(2, 1);
    }

    return bases;
}

torch::Tensor KANLinearImpl::curve2coeff(const torch::Tensor& grid, const torch::Tensor& noise) {
    // Placeholder: implement spline coefficient conversion
    return noise.permute({2,1,0}); // (out_features, in_features, grid_size + spline_order)
}

torch::Tensor KANLinearImpl::forward(const torch::Tensor& x) {
    auto base = torch::matmul(x, base_weight.transpose(0,1));
    auto spline = b_splines(x);

    auto sw = spline_weight;
    if (enable_standalone_scale_spline)
        sw = sw * spline_scaler.unsqueeze(-1);

    auto spline_out = torch::einsum("bif,oif->bo", {spline, sw});

    return base_activation(base) + spline_out;
}
/*torch::Tensor KANLinearImpl::curve2coeff(const torch::Tensor& x, const torch::Tensor& y) {
    // x: (num_points, in_features)
    // y: (num_points, in_features, out_features)
    // returns: (out_features, in_features, grid_size + spline_order)

    // Evaluate B-spline bases at the sample points
    auto bases = b_splines(x); // (num_points, in_features, K)
    int64_t num_points = bases.size(0);
    int64_t I = in_features;
    int64_t O = out_features;
    int64_t K = grid_size + spline_order;

    auto options = y.options();
    auto coeff = torch::zeros({O, I, K}, options);

    using namespace torch::indexing;

    for (int64_t i = 0; i < I; ++i) {
        // Design matrix B: (num_points, K)
        auto B = bases.index({Slice(), i, Slice()});          // (N, K)

        // Precompute normal matrix for stability: (K, K)
        auto Bt = B.transpose(0, 1);                          // (K, N)
        auto BtB = Bt.matmul(B);                              // (K, K)
        auto BtB_inv = torch::linalg_pinv(BtB);              // (K, K)
        auto pseudo_inv = BtB_inv.matmul(Bt);                 // (K, N)

        for (int64_t o = 0; o < O; ++o) {
            // Target curve: (num_points)
            auto y_io = y.index({Slice(), i, o});             // (N)

            // Solve least squares: c = (B^+ * y)
            auto c = pseudo_inv.matmul(y_io);                 // (K)

            coeff.index_put({o, i, Slice()}, c);
        }
    }

    return coeff; // (out_features, in_features, grid_size + spline_order)
}
*/
