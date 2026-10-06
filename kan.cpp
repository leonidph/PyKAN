#include "kan.h"



KANLayerImpl::KANLayerImpl(int in_f,int out_f,int knots)
    : in_features_(in_f), out_features_(out_f), num_knots_(knots),
      silu_(torch::nn::SiLU()) {

    register_module("silu", silu_);

    base_weight_ = register_parameter(
        "base_weight",
        torch::randn({out_features_, in_features_}) * 0.1);

    build_uniform_knots();

    int num_basis = num_knots_ + spline_degree_ - 1;

    spline_weight_ = register_parameter(
        "spline_weight",
        torch::randn({out_features_, in_features_, num_basis}) * 0.01);

    spline_scale_ = register_parameter(
        "spline_scale",
        torch::ones({out_features_, in_features_}));
}

void KANLayerImpl::build_uniform_knots() {
    std::vector<float> kv;
    for(int i=0;i<spline_degree_;++i) kv.push_back(-2.f);
    for(int i=0;i<num_knots_;++i) kv.push_back(-2.f + 4.f*i/(num_knots_-1));
    for(int i=0;i<spline_degree_;++i) kv.push_back(2.f);
    knots_ = torch::tensor(kv, torch::kFloat32);
}

void KANLayerImpl::refine_grid(const torch::Tensor& samples) {
    auto sorted = std::get<0>(samples.flatten().cpu().sort());
    std::vector<float> kv;

    float xmin = sorted[0].item<float>();
    float xmax = sorted[sorted.size(0)-1].item<float>();

    for(int i=0;i<spline_degree_;++i) kv.push_back(xmin);
    for(int i=0;i<num_knots_;++i){
        int idx=(sorted.size(0)-1)*i/(num_knots_-1);
        kv.push_back(sorted[idx].item<float>());
    }
    for(int i=0;i<spline_degree_;++i) kv.push_back(xmax);

    knots_ = torch::tensor(kv, torch::kFloat32);
}

torch::Tensor KANLayerImpl::bspline_basis(const torch::Tensor& x,int degree){
    auto knots = knots_.to(x.device());
    auto xe = x.unsqueeze(-1);

    int M = knots.size(0);
    int N0 = M - 1;

    std::vector<torch::Tensor> basis;
    for(int i=0;i<N0;++i)
        basis.push_back(((xe>=knots[i])&(xe<knots[i+1])).to(torch::kFloat32));

    for(int d=1; d<=degree; ++d){
        std::vector<torch::Tensor> next;
        for(int i=0;i<N0-d;++i){
            auto ld = knots[i+d]-knots[i];
            auto rd = knots[i+d+1]-knots[i+1];

            auto left=torch::zeros_like(basis[i]);
            auto right=torch::zeros_like(basis[i]);

            if(ld.item<float>()>1e-8f) left=((xe-knots[i])/ld)*basis[i];
            if(rd.item<float>()>1e-8f) right=((knots[i+d+1]-xe)/rd)*basis[i+1];

            next.push_back(left+right);
        }
        basis = std::move(next);
    }

    return torch::cat(basis,-1);
}

torch::Tensor KANLayerImpl::forward(torch::Tensor x){
    auto base_part = torch::matmul(silu_->forward(x), base_weight_.t());
    auto basis = bspline_basis(x, spline_degree_);
    auto scaled_weight = spline_weight_ * spline_scale_.unsqueeze(-1);
    auto spline_part = torch::einsum("bin,oin->bo", {basis, scaled_weight});
    return base_part + spline_part;
}
